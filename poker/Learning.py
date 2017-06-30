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


    __device__ float denominator = 0;
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
def forward(x):
    # W.T*x
    h1 = np.dot(model['W1'], x)
    # ReLU activation function
    h1[h1 < 0] = 0

    h2 = np.dot(model['w2'], h1)
    h2[h2 < 0] = 0

    y = np.dot(model['W2'], h2)
    return {
        'x': x,
        'h1': h1,
        'h2': h2,
        'y': y}


# TODO backward pass. DO IN CUDA
def backward(eph, epdlogp):
    pass

# ------------------------------------------------------------------
game = Game()
game.new_game()

num_inputs = 52  # TODO CALCULATE INPUT COUNT
num_outputs = 3  # CALL/CHECK, RAISE, FOLD
num_hiddens = [500]  # Each value represents number of nodes per layer

W1 = 0.1 * np.random.randn(num_inputs, num_hiddens[0]).astype(np.float32)
W2 = 0.1 * np.random.randn(num_hiddens[0], num_outputs).astype(np.float32)

model = {
    'W1': W1,
    'W2': W2,
}

forward_pass = mod.get_function("forward_pass")
softmax = mod.get_function("softmax")
input = game._bot1.get_hand().flatten().astype(np.float32)


W1_out = np.zeros((1, 500)).astype(np.float32)
y = np.zeros((1, 3)).astype(np.float32)
prediction = np.zeros((1, 500)).astype(np.float32)

input_gpu = cuda.mem_alloc(input.nbytes)
W1_gpu = cuda.mem_alloc(W1.nbytes)
W2_gpu = cuda.mem_alloc(W2.nbytes)
W1_out_gpu = cuda.mem_alloc(W1_out.nbytes)
y_gpu = cuda.mem_alloc(y.nbytes)
prediction_gpu = cuda.mem_alloc(prediction.nbytes)

cuda.memcpy_htod(input_gpu, input)
cuda.memcpy_htod(W1_gpu, W1)
cuda.memcpy_htod(W2_gpu, W2)
cuda.memcpy_htod(prediction_gpu, prediction)
cuda.memcpy_htod(y_gpu, y)


block = (16, 16, 1)
grid = ((500 + 16 - 1) / 16,
        (1 + 16 - 1) / 16)

print(grid)

forward_pass(input_gpu, W1_gpu, W1_out_gpu, np.int32(52), np.int32(500), np.int32(500), block=block, grid=grid)
forward_pass(W1_out_gpu, W2_gpu, y_gpu, np.int32(52), np.int32(500), np.int32(500), block=block, grid=grid)


cuda.memcpy_dtoh(y, y_gpu)
# DEBUG STATEMENT
print("Number of mistakes for matrix multiply: %d" % ((y - np.dot(input, W1)) > 0.001).sum())


softmax(np.int32(500), np.int32(1), y_gpu, prediction_gpu, block=block, grid=grid)

cuda.memcpy_dtoh(prediction, prediction_gpu)

# DEBUG STATEMENT
print("Number of mistakes for softmax: %d" % (prediction - softmax_cpu(y)).sum())
