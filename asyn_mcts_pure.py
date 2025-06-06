# -*- coding: utf-8 -*-

import numpy as np
import copy
from operator import itemgetter
from multiprocessing import Pool, cpu_count
from functools import partial


def rollout_task(state, limit=1000):
    """Standalone rollout for use in multiprocessing."""
    player = state.get_current_player()
    for i in range(limit):
        end, winner = state.game_end()
        if end:
            break
        action_probs = rollout_policy_fn(state)
        max_action = max(action_probs, key=itemgetter(1))[0]
        state.do_move(max_action)
    else:
        # If no break from the loop, issue a warning.
        print("WARNING: rollout reached move limit")
    if winner == -1:  # tie
        return 0
    else:
        return 1 if winner == player else -1


def callback_update(leaf_value, node, virtual_loss):
    node.revert_virtual_loss(virtual_loss)
    node.update_recursive(-leaf_value)


def rollout_policy_fn(board):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode:
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._W = 0.0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(
            self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct)
        )

    def add_virtual_loss(self, loss=1.0):
        self._n_visits += 1
        self._W -= loss

    def revert_virtual_loss(self, loss=1.0):
        self._n_visits -= 1
        self._W += loss

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update W, a running sum of values for all visits.
        self._W += leaf_value

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors."""
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        Q = self._W / self._n_visits if self._n_visits > 0 else 0
        return Q + u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS:
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(
        self, policy_value_fn, c_puct=5, n_playout=10000, virtual_loss=1.0, n_jobs=None
    ):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._virtual_loss = virtual_loss
        self.pool = Pool(processes=n_jobs or cpu_count())
        print(f"Using {self.pool._processes} processes for MCTS")

    def select_leaf(self, state):
        node = self._root
        state_copy = copy.deepcopy(state)

        while not node.is_leaf():
            action, next_node = node.select(self._c_puct)
            state_copy.do_move(action)
            node = next_node

        node.add_virtual_loss(self._virtual_loss)
        # Check for end of game
        end, winner = state_copy.game_end()
        if not end:
            action_probs, _ = self._policy(state_copy)
            node.expand(action_probs)

        return node, state_copy

    def get_move(self, state):
        """Runs all playouts asynchronously and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        pending = []

        for n in range(self._n_playout):
            node, state_copy = self.select_leaf(state)

            cb = partial(
                callback_update,
                node=node,
                virtual_loss=self._virtual_loss,
            )

            task = self.pool.apply_async(rollout_task, args=(state_copy,), callback=cb)

            pending.append(task)

            if len(pending) >= self.pool._processes * 2:
                pending[0].wait()
                pending.pop(0)

        for task in pending:
            task.wait()

        return max(
            self._root._children.items(),
            key=lambda act_node: act_node[1]._n_visits,
        )[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __del__(self):
        self.pool.terminate()
        self.pool.join()

    def __str__(self):
        return "MCTS"


class MCTSPlayer:
    """AI player based on MCTS"""

    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
