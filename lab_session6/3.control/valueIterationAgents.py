# valueIterationAgents.py
# -----------------------
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


import mdp, util
import numpy as np
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        super(ValueIterationAgent, self).__init__()

        import mdp as mdp_module
        self.mdp: mdp_module.MarkovDecisionProcess = mdp
        self.discount = discount
        print("using discount {}".format(discount))
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        delta = 0.01
        # TODO: Implement Value Iteration.
        for i in range(self.iterations):
            old_values = self.values.copy()
            for state in self.mdp.getStates():
                if len(self.mdp.getPossibleActions(state)) > 0:
                    self.values[state] = max(
                        [sum(
                            [prob * (self.mdp.getReward(state, action,
                                                        next_state)
                                     + (self.discount * old_values[next_state]))
                             for next_state, prob in
                             self.mdp.getTransitionStatesAndProbs(state,
                                                                  action)]
                        ) for action in self.mdp.getPossibleActions(state)]
                    )

            # Calculate Euclidean distance and break if not different enough
            if sum([
                x ** 2 for x in (self.values - old_values).values()
            ]) ** .5 < delta:
                print(f"iterations until convergence: {i}")
                break
        # Exit either when the number of iterations is reached,
        # OR until convergence (L2 distance < delta).
        # Print the number of iterations to convergence.

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # TODO: Implement this function according to the doc
        return sum([
            prob * (self.mdp.getReward(state, action, next_state)
                    + (self.discount * self.values[next_state]))
            for next_state, prob
            in self.mdp.getTransitionStatesAndProbs(state, action)])

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if len(self.mdp.getPossibleActions(state)) > 0:
            return max(self.mdp.getPossibleActions(state),
                       key=lambda action: sum([
                           prob * (self.mdp.getReward(state, action, next_state)
                                   + (self.discount * self.values[next_state]))
                           for next_state, prob
                           in
                           self.mdp.getTransitionStatesAndProbs(state, action)])
                       )
        else:
            return None
        # TODO: Implement according to the doc
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
