# -*- coding: utf-8 -*-

import numpy as np
import copy
from multiprocessing import Pool, cpu_count
from functools import partial

_model = None

def get_policy_value_fn():
    global _model
    if _model is None:
        from policy_value_net_pytorch import PolicyValueNet
        model_file = "models/best_policy_9_9_5.model"
        _model = PolicyValueNet(9, 9, model_file=model_file)
    return _model.policy_value_fn


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def inference_task(state):
    """Run neural net inference in subprocess."""
    policy_value_fn = get_policy_value_fn()
    action_probs, leaf_value = policy_value_fn(state)
    end, winner = state.game_end()
    if end:
        if winner == -1:
            leaf_value = 0.0
        else:
            leaf_value = 1.0 if winner == state.get_current_player() else -1.0
        action_probs = []
    return action_probs, leaf_value


def inference_callback(result, node, visited, virtual_loss):
    """Callback to update tree node statistics from async inference result."""
    action_probs, leaf_value = result
    for parent, action in reversed(visited):
        parent._children[action].revert_virtual_loss(virtual_loss)
    if action_probs:
        node.expand(action_probs)
    node.update_recursive(-leaf_value)


class TreeNode:
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
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
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
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
    """An implementation of Monte Carlo Tree Search with neural net inference in parallel."""

    def __init__(
        self, policy_value_fn, c_puct=5, n_playout=10000, virtual_loss=1.0, n_jobs=8
    ):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._virtual_loss = virtual_loss
        self.pool = Pool(processes=n_jobs or cpu_count())
        print(f"Using {self.pool._processes} processes for MCTS")

    def select_leaf(self, state):
        """Select a leaf node from the root and apply virtual loss."""
        node = self._root
        visited = []
        state_copy = copy.deepcopy(state)

        while not node.is_leaf():
            action, next_node = node.select(self._c_puct)
            visited.append((node, action))
            next_node.add_virtual_loss(self._virtual_loss)
            state_copy.do_move(action)
            node = next_node

        return node, state_copy, visited

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts asynchronously and return the available actions and their probabilities."""
        pending = []

        for n in range(self._n_playout):
            node, state_copy, visited = self.select_leaf(state)

            cb = partial(
                inference_callback,
                node=node,
                visited=visited,
                virtual_loss=self._virtual_loss,
            )

            task = self.pool.apply_async(
                inference_task, args=(state_copy,), callback=cb
            )
            pending.append(task)

            # if len(pending) >= self.pool._processes * 2:
            #     pending[0].wait()
            #     pending.pop(0)

        for task in pending:
            task.wait()

        act_visits = [
            (act, node._n_visits) for act, node in self._root._children.items()
        ]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

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

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75 * probs
                    + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))),
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
            #                location = board.move_to_location(move)
            #                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
