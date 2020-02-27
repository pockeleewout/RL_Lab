"""
Module containing the k-armed bandit problem
Complete the code wherever TODO is written.
Do not forget the documentation for every class and method!
We expect all classes to follow the Bandit abstract object formalism.
"""
# -*- coding: utf-8 -*-
import numpy as np
import typing


class Bandit(object):
    """
    Abstract concept of a Bandit, i.e. Slot Machine, the Agent can pull.

    A Bandit is a distribution over reals.
    The pull() method samples from the distribution to give out a reward.
    """

    def __init__(self, **kwargs):
        """
        Empty for our simple one-armed bandits, without hyperparameters.
        Parameters
        ----------
        **kwargs: dictionary
            Ignored additional inputs.
        """
        pass

    def reset(self):
        """
        Reinitializes the distribution.
        """
        pass

    def pull(self) -> float:
        """
        Returns a sample from the distribution.
        """
        raise NotImplementedError(
            "Calling method pull() in Abstract class Bandit")


class Gaussian_Bandit(Bandit):
    # TODO: implement this class following the formalism above.
    # Reminder: the Gaussian_Bandit's distribution is a fixed Gaussian.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mean = 0
        self.reset()

    def reset(self):
        self._mean = np.random.normal(loc=0, scale=1)

    def pull(self) -> float:
        return np.random.normal(loc=self._mean, scale=1)


class Gaussian_Bandit_NonStat(Bandit):
    # TODO: implement this class following the formalism above.
    # Reminder: the distribution mean changes each step over time,
    # with increments following N(m=0,std=0.01)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mean = 0
        self.reset()

    def reset(self):
        self._mean = np.random.normal(loc=0, scale=1)

    def pull(self) -> float:
        mean = self._mean
        self._mean += np.random.normal(loc=0, scale=0.01)
        return np.random.normal(loc=mean, scale=1)


def kbandit(bandit_class: typing.ClassVar):
    class KBandit(Bandit):
        # TODO: implement this class following the formalism above.

        def __init__(self, k: int = 1, **kwargs):
            super().__init__(**kwargs)
            self._bandits = [bandit_class() for _ in range(k)]
            self.reset()

        def reset(self):
            for bandit in self._bandits:
                bandit.reset()

        def pull(self, i: int = 0) -> float:
            return self._bandits[i].pull()

    return KBandit


KBandit = kbandit(Gaussian_Bandit)


KBandit_NonStat = kbandit(Gaussian_Bandit_NonStat)
