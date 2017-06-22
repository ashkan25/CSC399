import Hand
import Deck
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import Constants
import numpy as np

mod = SourceModule("""

    #define CUDA_KERNEL_LOOP(i, n) \
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
    
    template <typename Dtype>
    __global__ void forward(const int n, const Dtype* in, Dtype* out) {
        
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x) {
            out[i] = in[i] > 0 ? in[i] : 0;
        }
    }
  """)

num_inputs = 0  # TODO CALCULATE INPUT COUNT
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

f = open("HandRanks.dat", "rb")
lookup_table = np.fromfile(f, dtype=np.int32)

gpu_lookup_table = cuda.mem_alloc(lookup_table.nbytes)
cuda.memcpy_htod(gpu_lookup_table, lookup_table)
f.close()

game = Game()
game.new_game()
raw_input()
game.next_round()
raw_input()
game.next_round()
raw_input()
game.next_round()
