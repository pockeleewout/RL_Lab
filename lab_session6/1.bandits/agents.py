"""
Module containing the agent classes to solve a Bandit problem.

Complete the code wherever TODO is written.
Do not forget the documentation for every class and method!
An example can be seen on the Bandit_Agent and Random_Agent classes.
"""
# -*- coding: utf-8 -*-
import numpy as np
from utils import softmax, my_random_choice

class Bandit_Agent(object):
    """
    Abstract Agent to solve a Bandit problem.

    Contains the methods learn() and act() for the base life cycle of an agent.
    The reset() method reinitializes the agent.
    The minimum requirment to instantiate a child class of Bandit_Agent
    is that it implements the act() method (see Random_Agent).
    """
    def __init__(self, k:int, **kwargs):
        """
        Simply stores the number of arms of the Bandit problem.
        The __init__() method handles hyperparameters.
        Parameters
        ----------
        k: positive int
            Number of arms of the Bandit problem.
        kwargs: dictionary
            Additional parameters, ignored.
        """
        self.k = k

    def reset(self):
        """
        Reinitializes the agent to 0 knowledge, good as new.

        No inputs or outputs.
        The reset() method handles variables.
        """
        pass

    def learn(self, a:int, r:float):
        """
        Learning method. The agent learns that action a yielded reward r.
        Parameters
        ----------
        a: positive int < k
            Action that yielded the received reward r.
        r: float
            Reward for having performed action a.
        """
        pass

    def act(self) -> int:
        """
        Agent's method to select a lever (or Bandit) to pull.
        Returns
        -------
        a : positive int < k
            The action the agent chose to perform.
        """
        raise NotImplementedError("Calling method act() in Abstract class Bandit_Agent")

class Random_Agent(Bandit_Agent):
    """
    This agent doesn't learn, just acts purely randomly.
    Good baseline to compare to other agents.
    """
    def act(self):
        """
        Random action selection.
        Returns
        -------
        a : positive int < k
            A randomly selected action.
        """
        return np.random.randint(self.k)


class EpsGreedy_SampleAverage(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    # This class uses Sample Averages to estimate q; others are non stationary.

    def __init__(self, eps: float, **kwargs):
        super().__init__(**kwargs)
        self.q = np.zeros(self.k)
        self.eps = float(eps)
        self.steps = np.zeros(self.k)

    def reset(self):
        self.q = np.zeros(self.k)
        self.steps = np.zeros(self.k)

    def learn(self, a: int, r: float):
        self.q[a] += (r - self.q[a]) / self.steps[a]

    def act(self) -> int:
        if np.random.random() > self.eps:
            # Select best option
            act = max(enumerate(self.q), key=lambda q: q[1])[0]
        else:
            # Select random
            act = np.random.randint(self.k)
        self.steps[act] += 1
        return act


class EpsGreedy_WeightedAverage(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    # Non stationary agent with q estimating and eps-greedy action selection.

    def __init__(self, alpha: float, eps: float, **kwargs):
        super().__init__(**kwargs)
        self.q = np.zeros(self.k)
        self.alpha = float(alpha)
        self.steps = np.zeros(self.k)
        self.eps = float(eps)

    def reset(self):
        self.q = np.zeros(self.k)
        self.steps = np.zeros(self.k)

    def learn(self, a: int, r: float):
        self.q[a] = self.q[a] + self.alpha * (r - self.q[a])

    def act(self) -> int:
        if np.random.random() > self.eps:
            # Select best option
            act = max(enumerate(self.q), key=lambda q: q[1])[0]
        else:
            # Select random
            act = np.random.randint(self.k)
        self.steps[act] += 1
        return act
