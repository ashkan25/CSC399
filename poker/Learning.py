import Hand
import Deck
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import Constants
import numpy as np
import Game

# TODO Make sure you handle when matrix is bigger than NUM_BLOCKS*NUM_THREADS
mod = SourceModule("""
#include <stdio.h>

#define CUDA_KERNEL_LOOP(i, n) \
 for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

    __global__ void relu(const int width, const int height, const float* in, float* out) {

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if(row > height || col > width) {
            return;
        }

        out[row * width + col] = in[row * width + col] > 0 ? in[row * width + col] : 0;
    }


    __device__ float denominator;
    __global__ void softmax(const int width, const int height, const float* in, float* out) {

        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row >= height || col >= width) {
            return;
        }

        int index = col + row * width;

        atomicAdd(&denominator, expf(in[index]));

        out[index] = expf(in[index]) / denominator;

    }

    // https://www.shodor.org/media/content/petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
    __global__ void forward_pass(const float* a, const float* b, float* c,
                                const int a_width, const int b_width, const int c_width) {

        float c_value = 0.0;

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        // TODO pass in Height of matrix A. It is a vector, so it always should be 1
        if(row >= 1 || col >= b_width) {
            return;
        }

        for (int i = 0; i < a_width; i++) {
            c_value += (a[row * a_width + i]) * (b[i * b_width + col]);
        }

        c[row*c_width+col] = c_value;

        // TODO It might be better to export ReLU to another kernel since it is not needed for the final hidden layer
        // ReLU function: max(x, 0)
        //c[row * c_width + col] = c_value > 0 ? c_value  : 0;

    }

  """)


# FOR TESTING CPU VS GPU
def softmax_cpu(x):
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)


# FOR TESTING CPU VS GPU
# def forward(x):
#     # W.T*x
#     h1 = np.dot(model['W1'], x)
#     # ReLU activation function
#     h1[h1 < 0] = 0
#
#     h2 = np.dot(model['w2'], h1)
#     h2[h2 < 0] = 0
#
#     y = np.dot(model['W2'], h2)
#     return {
#         'x': x,
#         'h1': h1,
#         'h2': h2,
#         'y': y}


# TODO backward pass. DO IN CUDA
def backward_policy(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = eph.T.dot(epdlogp)
    dh = epdlogp.dot(model['W2'].T)
    dh[eph <= 0] = 0  # backpro prelu

    dW1 = epx.T.dot(dh)

    return {'W1': dW1, 'W2': dW2}



# Pick highest probability action, with a certain probability of picking a different choice
def pick_action(action_prob):
    r = np.random.uniform()
    total = 0
    for i, p in enumerate(action_prob[0]):
        total += p
        if r <= total:
            return i


# Compute discount rewards. Give more recent rewards more weight.
# TODO Check if it's possible to change function to work on GPU. Probably not, each value depends on the previous
def discount_rewards(rewards, discount_factor=0.98):
    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for i in reversed(xrange(0, len(rewards))):

        # Final state of round
        if rewards[i] != 0:
            running_add = 0

        running_add = running_add * discount_factor + rewards[i]
        discounted_r[i] = running_add
    return discounted_r


# ------------------------------------------------------------------
game = Game.Game()

num_inputs = np.int32(52)  # TODO CALCULATE INPUT COUNT
num_outputs = np.int32(3)  # CALL/CHECK, RAISE, FOLD
num_hiddens = [np.int32(500)]  # Each value represents number of nodes per layer
NUM_EPISODES = 1000
LEARNING_RATE = 1e-3
GAMMA = 0.99  # discount factor for reward
DECAY_RATE = 0.99  # decay factor for RMSProp leaky sum of grad^2
actions = []
rewards = []
model = {}
model['W1'] = 0.1 * np.random.randn(num_inputs, num_hiddens[0]).astype(np.float32)
model['W2'] = 0.1 * np.random.randn(num_hiddens[0], num_outputs).astype(np.float32)
grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}


def forward():
    forward_pass = mod.get_function("forward_pass")
    softmax = mod.get_function("softmax")
    input = game._bot1.get_hand().flatten().astype(np.float32)
    print('---')
    print(input)
    print('---')
    W1_out = np.zeros((1, num_hiddens[0])).astype(np.float32)
    y = np.zeros((1, num_outputs)).astype(np.float32)
    predictions = np.zeros((1, num_outputs)).astype(np.float32)

    input_gpu = cuda.mem_alloc(input.nbytes)
    W1_gpu = cuda.mem_alloc(model['W1'].nbytes)
    W2_gpu = cuda.mem_alloc(model['W2'].nbytes)
    W1_out_gpu = cuda.mem_alloc(W1_out.nbytes)
    y_gpu = cuda.mem_alloc(y.nbytes)
    predictions_gpu = cuda.mem_alloc(predictions.nbytes)

    cuda.memcpy_htod(input_gpu, input)
    cuda.memcpy_htod(W1_gpu, model['W1'])
    cuda.memcpy_htod(W2_gpu, model['W2'])
    cuda.memcpy_htod(y_gpu, y)
    cuda.memcpy_htod(predictions_gpu, predictions)

    block = (16, 16, 1)
    grid = ((500 + 16 - 1) / 16,
            (1 + 16 - 1) / 16)

    # print("BLOCK DIM: %s" % str(block))
    # print("GRID  DIM: %s" % str(grid))

    forward_pass(input_gpu, W1_gpu, W1_out_gpu, num_inputs, num_hiddens[0], num_hiddens[0], block=block, grid=grid)
    forward_pass(W1_out_gpu, W2_gpu, y_gpu, num_hiddens[0], num_outputs, num_outputs, block=block, grid=grid)

    # TODO remove if y is not used
    cuda.memcpy_dtoh(y, y_gpu)

    cuda.memcpy_dtoh(W1_out, W1_out_gpu)

    # DEBUG STATEMENT
    # print("Number of mistakes for matrix multiply (Hidden): %d" % (np.abs(W1_out - np.dot(input, W1)) > 0.001).sum())
    # print("Number of mistakes for matrix multiply (Output): %d" % (np.abs(y - np.dot(W1_out, W2)) > 0.001).sum())

    # Work around for denominator that needs to be reset for each round
    denom = np.array([0]).astype(np.float32)
    denom_gpu, _ = mod.get_global("denominator")
    cuda.memcpy_htod(denom_gpu, denom)

    softmax(num_outputs, np.int32(1), y_gpu, predictions_gpu, np.int32(0), block=block, grid=grid)

    cuda.memcpy_dtoh(predictions, predictions_gpu)

    # DEBUG STATEMENT
    # print("Number of mistakes for softmax: %d" % (np.abs(prediction - softmax_cpu(y)) > 0.001).sum())
    print("Prediction probabilities: %s" % str(predictions))

    return predictions, W1_out


def next_round():
    raw_input()
    game.next_round()
    forward()


for i in range(NUM_EPISODES):
    game.new_game()

    xs, hs, dlogps, rewards = [], [], [], []

    for _ in range(3):
        action_prob, h = forward()
        action = pick_action(action_prob)
        actions.append(action)
        next_round()
        # No rewards for first 3 phases
        rewards.append(0)
        hs.append(h)

        # softmax loss gradient
        dlogsoftmax = action_prob
        dlogsoftmax[0, action] -= 1
        dlogps.append(dlogsoftmax)

        # Observation/input
        xs.append(game.get_p1_hand())

    action_prob, h = forward()
    action = pick_action(action_prob)
    actions.append(action)

    reward = game.evaluate_winner(game.get_p1_hand(), game.get_p2_hand())

    rewards.append(reward)  # +1 / -1 depending on who wins. 0 for tie
    hs.append(h)

    rewards = discount_rewards(rewards)

    # standardize the rewards
    # Only 10 rewards, not worth it to do on GPU.
    rewards -= np.mean(rewards)
    rewards /= np.std(rewards)

    # Cannot be done on GPU, order must be preserved
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(rewards)

    grad = backward(eph, epdlogp)

    # TODO Check if it is worth it to do operation in parallel. 2 Matrices is 500x4, approx 40x40 matrix.
    for k in model:
        # accumulate grad over batch
        grad_buffer[k] += grad[k]

    if i % 10 == 0:
        for k, v in model.iteritems():
            g = grad_buffer[k]  # gradient
            rmsprop_cache[k] = DECAY_RATE * rmsprop_cache[k] + (1 - DECAY_RATE) * g ** 2
            model[k] -= LEARNING_RATE * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v)

