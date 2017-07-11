from __future__ import division
import Hand
import Deck
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import Constants
import numpy as np
import Game
import math

# TODO Make sure you handle when matrix is bigger than NUM_BLOCKS*NUM_THREADS
mod = SourceModule("""
#include <stdio.h>

#define CUDA_KERNEL_LOOP(i, n) \
 for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

    __global__ void relu(const int width, const int height, float* a) {

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if(row >= height || col >= width) {
            return;
        }

        int index = col + row * width;

        if (a[index] < 0) {
            a[index] = 0;
        }
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
    
    #define TILE_DIM 32
    #define BLOCK_ROWS 8
    #define nreps 1
    __global__ void matrix_transpose(int width, int height, float *in, float *out) {
        __shared__ float block[TILE_DIM][TILE_DIM];
        int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
        int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

        int index = xIndex + (yIndex*width);
 
        xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
        yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
        int index_out = xIndex + (yIndex*height); 

        for (int r=0; r<nreps; r++) {
            for (int i=0; i < TILE_DIM; i += BLOCK_ROWS) {
                block[threadIdx.y+i][threadIdx.x] =
                in[index+i*width];
            }

            __syncthreads();
            for (int i=0; i < TILE_DIM; i += BLOCK_ROWS) {
                out[index_out+(i*height)] =block[threadIdx.x][threadIdx.y+i];
            }
        }   
    }

    // TODO optimize using shared memory
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
def backward_policy(eph, epdlogp, epx):
    dW2 = eph.T.dot(epdlogp)

    dh = epdlogp.dot(model['W2'].T)
    dh[eph <= 0] = 0

    dW1 = epx.T.dot(dh)

    return {'W1': dW1, 'W2': dW2}


# Pick highest probability action, with a certain probability of picking a different choice
def pick_action(action_prob):
    r = np.random.uniform()
    total = 0
    print(action_prob)
    for i, p in enumerate(action_prob[0]):
        total += p
        if r <= total:
            return i

    # action_prob should always sum to 1. This is in case of small rounding error
    return i

# Compute discount rewards. Give more recent rewards more weight.
# TODO Check if it's possible to change function to work on GPU. Probably not, each value depends on the previous
def discount_rewards(rewards, discount_factor=0.98):
    discounted_r = np.zeros_like(rewards).astype(np.float32)
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

num_inputs = np.int32(52*3)  # TODO CALCULATE INPUT COUNT (Make 52x3, 52 for bets, 52 for round, 52 for hand)
num_outputs = np.int32(3)  # CALL/CHECK, RAISE, FOLD
num_hiddens = [np.int32(1000)]  # Each value represents number of nodes per layer
NUM_EPISODES = 100000
LEARNING_RATE = 1e-3
GAMMA = 0.99  # discount factor for reward
DECAY_RATE = 0.99  # decay factor for RMSProp leaky sum of grad^2
actions = []
reward_count = []
model = {}
model['W1'] = 0.1 * np.random.randn(num_inputs, num_hiddens[0]).astype(np.float32)
model['W2'] = 0.1 * np.random.randn(num_hiddens[0], num_outputs).astype(np.float32)
grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}


def forward():
    # input = np.array([1, 2, 3, 4]).flatten().astype(np.float32)

    # input = game._bot1.get_hand().flatten().astype(np.float32)
#    hand_input = game._bot1.get_hand()
#    round_input = game.get_round_input()
#    bet_input = game.get_bet_input()
#   input = np.concatenate((hand_input, bet_input, round_input)).astype(np.float32)
    input = game.get_input() 

   # print(input)
    # print(input.shape)


    forward_pass = mod.get_function("forward_pass")
    softmax = mod.get_function("softmax")
    relu = mod.get_function("relu")

    # print('---')
    # print(input)
    # print('---')

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

    block_x, block_y = 16, 16
    block = (block_x, block_y, 1)
    grid = (int(math.ceil(model['W1'].shape[1] / block_x)),
            int(math.ceil(input.shape[0] / block_x)))

    #print("BLOCK DIM: %s" % str(block))
    #print("GRID  DIM: %s" % str(grid))

    forward_pass(input_gpu, W1_gpu, W1_out_gpu, num_inputs, num_hiddens[0], num_hiddens[0], block=block, grid=grid)

    # DEBUG
    # debug_answer = np.dot(input, model['W1'])
    # cuda.memcpy_dtoh(W1_out, W1_out_gpu) # TODO REMOVE IF NOT DEBUG
    # print("Number of mistakes for matrix multiply (Hidden 1): %d" % (np.abs(W1_out - debug_answer) > 0.001).sum())

    relu(num_hiddens[0], np.int32(1), W1_out_gpu, block=block, grid=grid)

    cuda.memcpy_dtoh(W1_out, W1_out_gpu)

    # DEBUG
    # debug_answer[debug_answer < 0] = 0 # RELU
    # print("Number of mistakes for RELU (Hidden 1): %d" % (np.abs(W1_out - debug_answer) > 0.001).sum())

    block_x, block_y = 16, 16
    block = (block_x, block_y, 1)
    grid = (int(math.ceil(model['W2'].shape[1] / block_x)),
            int(math.ceil(W1_out.shape[1] / block_x)))


    #print(grid)
    #print(block)
    forward_pass(W1_out_gpu, W2_gpu, y_gpu, num_hiddens[0], num_outputs, num_outputs, block=block, grid=grid)

    # DEBUG
    # cuda.memcpy_dtoh(y, y_gpu)
    # debug_answer = np.dot(debug_answer, model['W2'])
    # print("Number of mistakes for matrix multiply (Output): %d" % (np.abs(y - debug_answer) > 0.001).sum())

    # Work around for denominator that needs to be reset for each round
    denom = np.array([0]).astype(np.float32)
    denom_gpu, _ = mod.get_global("denominator")
    cuda.memcpy_htod(denom_gpu, denom)

    softmax(num_outputs, np.int32(1), y_gpu, predictions_gpu, np.int32(0), block=block, grid=grid)

    cuda.memcpy_dtoh(predictions, predictions_gpu)

    # DEBUG STATEMENT
    # print("Number of mistakes for softmax: %d" % (np.abs(predictions - softmax_cpu(y)) > 0.001).sum())
    # print("Prediction probabilities: %s" % str(predictions))

    return predictions, W1_out

def backward():
    a = np.random.random((64, 512)).astype(np.float32)
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)
    transpose = mod.get_function("matrix_transpose")

    b = np.zeros((512, 64)).astype(np.float32)
    b_gpu = cuda.mem_alloc(b.nbytes)
    cuda.memcpy_htod(b_gpu, b)
    

    block = (32, 8, 1)
#    grid = (64/32, 512/32) 
    grid = (2, 16)

    transpose(np.int32(512), np.int32(64), a_gpu, b_gpu, block=block, grid=grid)

    cuda.memcpy_dtoh(b, b_gpu)
    print(b[2])
    print(a.T[2])
    print("NUMBER OF MISTAKES: %d" % (np.abs(b-a.T) > 0.001).sum())


def next_round():
    # raw_input()
    game.next_round()
    # forward()


def update_learning_params(xs, hs, dlogps, rewards, action_raise=False):
    action_prob, h = forward()

    action = pick_action(action_prob)

    # actions.append(action)

    # Raise only once. If the chosen action is to raise again, check instead.
    if Constants.ACTIONS[action] == "RAISE" and action_raise:
        action = 0

    hs.append(h)

    # https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
    dlogsoftmax = action_prob
    dlogsoftmax[0, action] -= 1
    dlogps.append(dlogsoftmax)

    # Observation/input
    # xs.append(game._bot1.get_hand().flatten().astype(np.float32))
    xs.append(game.get_input())

    return action


# Allow only one Raise per state. Return True if action for either player was FOLD
def handle_action(action, rewards):
    # bot2_action = game.bot_decision(bot_can_raise)

    if Constants.ACTIONS[action] == "FOLD":
        rewards.append(-1 * game.get_num_bets())
        return True
    elif Constants.ACTIONS[action] == "RAISE":
        # ALREADY RAISED, why making another action?
        # action = update_learning_params(xs, hs, dlogps, rewards, action_raise=True)
        # rewards.append(0)
        bot2_action = game.bot_decision(can_raise=False)
        # return handle_action(action, rewards, bot_can_raise=False)
    else:
        bot2_action = game.bot_decision(can_raise=True)

    if Constants.ACTIONS[bot2_action] == "FOLD":
        rewards.append(1 * game.get_num_bets())
        return True
    elif Constants.ACTIONS[bot2_action] == "RAISE":
        action = update_learning_params(xs, hs, dlogps, rewards, action_raise=True)
        rewards.append(0)
        if Constants.ACTIONS[action] == "FOLD":
            rewards.append(-1 * game.get_num_bets())
            return True

    if Constants.ACTIONS[bot2_action] == "RAISE" or Constants.ACTIONS[action] == "RAISE":
        game.add_bet()

    # Otherwise, both players have checked
    rewards.append(0)
    return False



for i in range(NUM_EPISODES):
    game.new_game()

    xs, hs, dlogps, rewards = [], [], [], []

    for _ in range(3):

        action = update_learning_params(xs, hs, dlogps, rewards)
        is_fold = handle_action(action, rewards)

        if is_fold:
            break

        next_round()


    backward()
    break


    if not is_fold:
        action = update_learning_params(xs, hs, dlogps, rewards)
        is_fold = handle_action(action, rewards)

    if not is_fold:
        # Remove the last reward. The length of rewards must be equal to the number of actions
        rewards.pop()
        # EVALUATE WINNER
        reward = game.evaluate_winner(game.get_p1_hand(), game.get_p2_hand())

        rewards.append(reward * game.get_num_bets())  # +1 / -1 depending on who wins. 0 for tie

    rewards = discount_rewards(rewards)
    # print(rewards)

    # DEBUG
    reward_count.append(rewards[-1])

    # standardize the rewards
    # (Special Case) Don't standardize if someone folds at the start (preflop)
    #    if len(rewards) != 1:
    #        rewards -= np.mean(rewards).astype(np.float32)
    #        rewards /= np.std(rewards).astype(np.float32)

    # Cannot be done on GPU, order must be preserved

    if i % 100 == 0:

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(rewards)
        epdlogp *= epr

        grad = backward_policy(eph, epdlogp, epx)

        # TODO Check if it is worth it to do operation in parallel. 2 Matrices is 500x4, approx 40x40 matrix. Input size will get larger in the future
        for k in model:
            # accumulate grad over batch
            grad_buffer[k] += grad[k]

        if i % 1000 == 0:
            for k, v in model.iteritems():
                g = grad_buffer[k]  # gradient

                # RMSprop: Gradient descent optimization algorithms
                rmsprop_cache[k] = DECAY_RATE * rmsprop_cache[k] + (1 - DECAY_RATE) * g ** 2

                # Update weights to minimize the error
                model[k] -= LEARNING_RATE * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)

                # Reset grad buffer
                grad_buffer[k] = np.zeros_like(v)

                #    if i%10 == 0:
                #        print(rewards)

    if (i > 0 and i % 2000 == 0):
        x = np.array(reward_count)
        unique, counts = np.unique(x, return_counts=True)
        values = np.asarray((unique, counts)).T
        earning = 0
        for i in range(values.shape[0]):
            earning += values[i][1] * values[i][0]
        print(earning)
        reward_count = []

x = np.array(reward_count)
unique, counts = np.unique(x, return_counts=True)

values = np.asarray((unique, counts)).T
print(values)
earning = 0
for i in range(11):
    earning += values[i][1] * values[i][0]
loss = 0
for i in range(0, 5):
    loss += values[i][1] * values[i][0]
win = 0
for i in range(6, 11):
    win += values[i][1] * values[i][0]
print(earning)
print(win)
print(loss)
