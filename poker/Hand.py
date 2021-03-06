import Constants
import numpy as np


# ----------------------------------------
# HAND FORMAT: 4x13 array
# -23456789TJQKA
# C.............
# D.............
# H.............
# S.............
#
# First index represents the SUIT: 0 : CLUBS, 1 : DIAMONDS, 2 : HEARTS, 3 : SPADES
# second index represents the VALUE - 0 : 1, ... , 8 : 9, 9 : 10, 10 : JACK, 11 : QUEEN, 12 : KING, 13 : ACE
#
# Example: 2 of Hearts = [2*NUM_OF_VALUES + 0], Jack of Spades = [3*NUM_OF_VALUES + 9]
# ----------------------------------------

class Hand:
    def __init__(self):
        self._hand = np.zeros(Constants.NUM_OF_VALUES*Constants.NUM_OF_SUITS, dtype=np.float32)
        self._hand_lt_format = []

    def reset_hand(self):
        self._hand = np.zeros(Constants.NUM_OF_VALUES*Constants.NUM_OF_SUITS, dtype=np.float32)
        self._hand_lt_format = []

    def add_card(self, cards):
        for card in cards:
            self._hand[card[0]*Constants.NUM_OF_VALUES + card[1]] = 1
            self._hand_lt_format.append((card[0] + 1) + (card[1] * Constants.NUM_OF_SUITS))

    def __str__(self):
        return str(self._hand)

    def get_hand(self):
        return self._hand

    def get_lookup_table_hand(self):
        return self._hand_lt_format
