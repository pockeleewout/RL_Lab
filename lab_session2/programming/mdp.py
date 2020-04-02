# mdp.py
# ------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random
from typing import *


class MarkovDecisionProcess:

    def getStates(self) -> List[Tuple[int, int]]:
        """
        Return a list of all states in the MDP.
        Not generally possible for large MDPs.
        """
        raise NotImplementedError()

    def getStartState(self) -> Tuple[int, int]:
        """
        Return the start state of the MDP.
        """
        raise NotImplementedError()

    def getPossibleActions(self, state) -> Tuple[str]:
        """
        Return list of possible actions from 'state'.
        """
        raise NotImplementedError()

    def getTransitionStatesAndProbs(self, state, action) -> List[Tuple[Tuple[int, int], float]]:
        """
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.

        Note that in Q-Learning and reinforcment
        learning in general, we do not know these
        probabilities nor do we directly model them.
        """
        raise NotImplementedError()

    def getReward(self, state, action, nextState) -> Union[int, float]:
        """
        Get the reward for the state, action, nextState transition.

        Not available in reinforcement learning.
        """
        raise NotImplementedError()

    def isTerminal(self, state) -> bool:
        """
        Returns true if the current state is a terminal state.  By convention,
        a terminal state has zero future rewards.  Sometimes the terminal state(s)
        may have no possible actions.  It is also common to think of the terminal
        state as having a self-loop action 'pass' with zero reward; the formulations
        are equivalent.
        """
        raise NotImplementedError()
