class Character:
    def __init__(self, firstGuessClass, firstGuess, firstGuessConfidentLvl):
        self._firstGuessClass = firstGuessClass
        self._firstGuess = firstGuess
        self._firstGuessConfidentLvl = firstGuessConfidentLvl

    @property
    def firstGuessClass(self):
        return self._firstGuessClass

    @firstGuessClass.setter
    def firstGuessClass(self, firstGuessClass):
        self._firstGuessClass = firstGuessClass

    @property
    def firstGuess(self):
        return self._firstGuess

    @firstGuess.setter
    def firstGuess(self, firstGuess):
        self._firstGuess = firstGuess

    @property
    def firstGuessConfidentLvl(self):
        return self._firstGuessConfidentLvl

    @firstGuessConfidentLvl.setter
    def firstGuessConfidentLvl(self, firstGuessConfidentLvl):
        self._firstGuessConfidentLvl = firstGuessConfidentLvl

