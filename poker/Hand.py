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
# Example: 2 of Hearts = [2][0], Jack of Spades = [3][9]
# ----------------------------------------

class Hand:
    def __init__(self):
        # TODO Change hand representation to 1D array of size 52 to remove the need to flatten array every time
        self._hand = np.zeros((Constants.NUM_OF_SUITES, Constants.NUM_OF_VALUES))

    def reset_hand(self):
        # TODO DO ON GPU
        self._hand = np.zeros((Constants.NUM_OF_SUITES, Constants.NUM_OF_VALUES))

    def add_card(self, cards):
        for card in cards:
            self._hand[card[0]][card[1]] = 1

    def __str__(self):
        return str(self._hand)

    def get_hand(self):
	return self._hand
