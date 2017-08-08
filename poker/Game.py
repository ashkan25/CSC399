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
        self._bet = 1
        self._round_input = np.zeros(52, dtype=np.float32)
        self._bet_input = np.array([1 if i % 13 == 0 or i % 13 == 1 else 0 for i in range(52)], dtype=np.float32)
        self._bot_cumsum_prob = list(np.cumsum(Constants.BOT_ACTION_PROBS))

        f = open("HandRanks.dat", "rb")
        self._lookup_table = np.fromfile(f, dtype=np.int32)
        f.close()

    def new_game(self):
        self._bot1 = Hand.Hand()
        self._bot2 = Hand.Hand()
        self._deck = Deck.Deck()
        self._round = 0
        self._bet = 1
        self._round_input = np.zeros(52, dtype=np.float32)
        self._bet_input = np.array([1 if i % 13 == 0 or i % 13 == 1 else 0 for i in range(52)], dtype=np.float32)

        # PREFLOP
        self._bot1.add_card(self._deck.draw(2))
        self._bot2.add_card(self._deck.draw(2))

        self.round_input_update()
        self._round += 1

    def get_input(self):
        return np.concatenate((self._bot1.get_hand(), self._bet_input, self._round_input)).astype(np.float32) 

    def get_input_P2(self):
        return np.concatenate((self._bot2.get_hand(), self._bet_input, self._round_input)).astype(np.float32)

    def next_round(self):
        if self._round == Constants.FLOP:
            flop_cards = self._deck.draw(3)
        else:
            flop_cards = self._deck.draw()

        self._bot1.add_card(flop_cards)
        self._bot2.add_card(flop_cards)
        self.round_input_update()
        self._round += 1

    def get_num_bets(self):
        return self._bet

    def round_input_update(self):
        for i in range(Constants.NUM_OF_SUITS):
            self._round_input[self._round*3 + i*Constants.NUM_OF_VALUES] = 1
            self._round_input[self._round*3 + 1 + i*Constants.NUM_OF_VALUES] = 1
            self._round_input[self._round*3 + 2 + i*Constants.NUM_OF_VALUES] = 1

    def get_round_input(self):
        return self._round_input

    def get_bet_input(self):
        return self._bet_input

    def add_bet(self):
        for i in range(Constants.NUM_OF_SUITS):
            self._bet_input[self._bet*2 + i*13] = 1
            self._bet_input[self._bet*2 + 1 + i*13] = 1
        self._bet += 1

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
        return i