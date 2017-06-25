import Hand
import Deck
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import Constants
import numpy as np


# TODO Make sure you handle when matrix is bigger than NUM_BLOCKS*NUM_THREADS
mod = SourceModule("""
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
        
        if (row > height || col > width) {
            return;
        }
        
        int index = col + row * width;
        atomicAdd(&denominator, expf(in[index]));
        
        // TODO Check if sync is needed. It only syncs for individual blocks, but atomic should sync for all blocks
        __syncthreads(); 
        
        out[index] = in[index] / denominator;
        
    }
    
    // https://www.shodor.org/media/content/petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
    __global__ void forward_pass(const float* a, const float* b, float* c,
                                const int a_width, const int b_width, const int c_width) {
    
        float c_value = 0.0;
    
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if(row > a_width || col > b_width) {
            return;
        }
        
        for (int i = 0; i < a_width; i++) {
            c_value += (a[row * a_width + i]) * (b[i * b_width + col]);
        }
        
        // TODO It might be better to export ReLU to another kernel since it doesn't 
        // ReLU function: max(x, 0)
        c[row * c_width + col] = c_value > 0 ? c_value  : 0;
    
    }

  """)

num_inputs = 52  # TODO CALCULATE INPUT COUNT
num_outputs = 7
num_hiddens = [500, 500]

W1 = 0.1 * np.random.randn(num_inputs, num_hiddens[0])
W2 = 0.1 * np.random.randn(num_hiddens[0], num_hiddens[1])
W3 = 0.01 * np.random.randn(num_hiddens[1], num_outputs)

model = {
    'W1': W1,
    'W2': W2,
    'W3': W3,
}


# TODO Convert to CUDA
def Softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)


# TODO Convert to CUDA
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

class Game:
    def __init__(self):
        self._bot1 = Hand.Hand()
        self._bot2 = Hand.Hand()
        self._deck = Deck.Deck()
        self._round = 0

    def new_game(self):

        # PREFLOP
        self._bot1.add_card(self._deck.draw(2))
        self._bot2.add_card(self._deck.draw(2))

        print(Constants.ROUNDS[self._round])
        print(self._bot1)
        print(self._bot2)
        self._round += 1

    def next_round(self):
        if self._round == Constants.FLOP:
            flop_cards = self._deck.draw(3)
        else:
            flop_cards = self._deck.draw()

        self._bot1.add_card(flop_cards)
        self._bot2.add_card(flop_cards)
        print(Constants.ROUNDS[self._round])
        print(self._bot1)
        print(self._bot2)
        self._round += 1


# lookup_table = array.array('i')
# f = open("HandRanks.dat", "rb")
# lookup_table.fromfile(f, 32487834)
# print(len(lookup_table))
# f.close()
# # Needs to be casted to numpy array. Python array does not have buffer interface
# lookup_table = np.array(lookup_table)

#f = open("HandRanks.dat", "rb")
#lookup_table = np.fromfile(f, dtype=np.int32)

#gpu_lookup_table = cuda.mem_alloc(lookup_table.nbytes)
#cuda.memcpy_htod(gpu_lookup_table, lookup_table)
#f.close()

game = Game()
game.new_game()
raw_input()
game.next_round()
raw_input()
game.next_round()
raw_input()
game.next_round()


forward_pass = mod.get_function("forward_pass")
input = game._bot1.get_hand().flatten()

input = input.astype(np.float32)
W1 = W1.astype(np.float32)

print(input.shape)
print(W1.shape)

input_gpu = cuda.mem_alloc(input.nbytes)
cuda.memcpy_htod(input_gpu, input)

W1_gpu = cuda.mem_alloc(W1.nbytes)
cuda.memcpy_htod(W1_gpu, W1)

forward_pass()
