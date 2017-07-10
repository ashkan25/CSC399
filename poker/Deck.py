import Constants
import numpy as np


class Deck:
    def __init__(self):
        self._deck = self._setup()

    def _setup(self):
        cards = np.empty(shape=(Constants.NUM_OF_SUITS*Constants.NUM_OF_VALUES, 2))
        for i in range(Constants.NUM_OF_SUITS):
            for j in range(Constants.NUM_OF_VALUES):
                cards[(i*Constants.NUM_OF_VALUES) + j] = [i,j]
        np.random.shuffle(cards)
        return cards.astype(int)

    def draw(self, n=1):
        cards = self._deck[:n]
        self._deck = self._deck[n:]
        return cards
