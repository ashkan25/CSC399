import Hand
import Deck
import Constants
import numpy as np


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

# f = open("HandRanks.dat", "rb")
# lookup_table = np.fromfile(f, dtype=np.int32)

# gpu_lookup_table = cuda.mem_alloc(lookup_table.nbytes)
# cuda.memcpy_htod(gpu_lookup_table, lookup_table)
# f.close()

if __name__ == '__main__':
    game = Game()
    game.new_game()
    raw_input()
    game.next_round()
    raw_input()
    game.next_round()
    raw_input()
    game.next_round()
