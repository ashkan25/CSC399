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

        self._bot_cumsum_prob = np.cumsum(Constants.BOT_ACTION_PROBS)

        f = open("HandRanks.dat", "rb")
        self._lookup_table = np.fromfile(f, dtype=np.int32)
        f.close()

    def new_game(self):
        self._bot1 = Hand.Hand()
        self._bot2 = Hand.Hand()
        self._deck = Deck.Deck()
        self._round = 0

        # PREFLOP
        self._bot1.add_card(self._deck.draw(2))
        self._bot2.add_card(self._deck.draw(2))

#        print(Constants.ROUNDS[self._round])
#        print(self._bot1)
#        print(self._bot2)
        self._round += 1

    def next_round(self):
        if self._round == Constants.FLOP:
            flop_cards = self._deck.draw(3)
        else:
            flop_cards = self._deck.draw()

        self._bot1.add_card(flop_cards)
        self._bot2.add_card(flop_cards)
        print(Constants.ROUNDS[self._round])
#        print(self._bot1)
#        print(self._bot2)
        self._round += 1

    def hand_strength(self, h):

        p = self._lookup_table[53 + h[0]]
        p = self._lookup_table[p + h[1]]
        p = self._lookup_table[p + h[2]]
        p = self._lookup_table[p + h[3]]
        p = self._lookup_table[p + h[4]]
        p = self._lookup_table[p + h[5]]
        return self._lookup_table[p + h[6]]

    def evaluate_winner(self, h1, h2):
        p1 = self.hand_strength(h1)
        p2 = self.hand_strength(h2)

        print("P1 HAND STRENGTH: %d" % p1)
        print("P2 HAND STRENGTH: %d" % p2)

        if p1 > p2:
            return 1
        elif p2 > p1:
            return -1
        else:
            return 0

    def get_p1_hand(self):
        return self._bot1.get_lookup_table_hand()

    def get_p2_hand(self):
        return self._bot2.get_lookup_table_hand()

    def bot_decision(self, can_raise=True):

        # Remove raise as an option if bot cannot raise (Case when someone already raised)
        if can_raise:
            probs = self._bot_cumsum_prob
        else:
            probs = self._bot_cumsum_prob[:]
            probs[0] = probs.pop(1)

        r = np.random.uniform()
        for i, v in enumerate(probs):
            if r < v:
                return i

# lookup_table = array.array('i')
# f = open("HandRanks.dat", "rb")
# lookup_table.fromfile(f, 32487834)
# print(len(lookup_table))
# f.close()
# # Needs to be casted to numpy array. Python array does not have buffer interface
# lookup_table = np.array(lookup_table)

# gpu_lookup_table = cuda.mem_alloc(lookup_table.nbytes)
# cuda.memcpy_htod(gpu_lookup_table, lookup_table)


if __name__ == '__main__':
    game = Game()
    game.new_game()
    raw_input()
    game.next_round()
    raw_input()
    game.next_round()
    raw_input()
    game.next_round()
    print(game.evaluate_winner(game.get_p1_hand(), game.get_p2_hand()))
    print(game.get_p1_hand())
    print(game.get_p2_hand())
