#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Callable, List

import toh_mdp as tm


def value_iteration(
        mdp: tm.TohMdp, v_table: tm.VTable
) -> Tuple[tm.VTable, tm.QTable, float]:
    """Computes one step of value iteration.

    Hint 1: Since the terminal state will always have value 0 since
    initialization, you only need to update values for nonterminal states.

    Hint 2: It might be easier to first populate the Q-value table.

    Args:
        mdp: the MDP definition.
        v_table: Value table from the previous iteration.

    Returns:
        new_v_table: tm.VTable
            New value table after one step of value iteration.
        q_table: tm.QTable
            New Q-value table after one step of value iteration.
        max_delta: float
            Maximum absolute value difference for all value updates, i.e.,
            max_s |V_k(s) - V_k+1(s)|.
    """
    new_v_table: tm.VTable = v_table.copy()
    q_table: tm.QTable = {}
    # noinspection PyUnusedLocal
    max_delta = 0.0
    # *** BEGIN OF YOUR CODE ***
    # Pre-assign states, actions, transition, reward, discount
    nonterminal_states = mdp.nonterminal_states
    all_states = mdp.all_states
    actions = mdp.actions
    trans = mdp.transition
    reward = mdp.reward
    gamma = mdp.config.gamma

    # Iterate through each state, action, and new state to update Q-value using Bellman eqns
    for s in nonterminal_states:  # state
        new_v = -10000
        for a in actions:  # action
            new_q = 0.0
            for sprime in all_states:  # new state s'
                new_q += trans(s,a,sprime)*(reward(s,a,sprime)+gamma*v_table[sprime])  # Bellman

            q_table[(s, a)] = new_q
            new_v = max(new_v, new_q)

        max_s = abs(v_table[s]-new_v)
        max_delta = max(max_delta, max_s)
        new_v_table[s] = new_v

    # ***  END OF YOUR CODE  ***
    return new_v_table, q_table, max_delta


def extract_policy(
        mdp: tm.TohMdp, q_table: tm.QTable
) -> tm.Policy:
    """Extract policy mapping from Q-value table.

    Remember that no action is available from the terminal state, so the
    extracted policy only needs to have all the nonterminal states (can be
    accessed by mdp.nonterminal_states) as keys.

    Args:
        mdp: the MDP definition.
        q_table: Q-Value table to extract policy from.

    Returns:
        policy: tm.Policy
            A Policy maps nonterminal states to actions.
    """
    # *** BEGIN OF YOUR CODE ***
    policy = {}  # initialize dictionary
    for (s, a) in q_table.keys():
        if s not in policy.keys():
            policy[s] = a
        if s is mdp.goal:
            policy[s] = "Exit"
        else:
            if q_table[(s, a)] > q_table[(s, policy[s])]:
                policy[s] = a
    return policy


def q_update(
        mdp: tm.TohMdp, q_table: tm.QTable,
        transition: Tuple[tm.TohState, tm.TohAction, float, tm.TohState],
        alpha: float) -> None:
    """Perform a Q-update based on a (S, A, R, S') transition.

    Update the relevant entries in the given q_update based on the given
    (S, A, R, S') transition and alpha value.

    Args:
        mdp: the MDP definition.
        q_table: the Q-Value table to be updated.
        transition: A (S, A, R, S') tuple representing the agent transition.
        alpha: alpha value (i.e., learning rate) for the Q-Value update.
    """
    state, action, reward, next_state = transition
    # *** BEGIN OF YOUR CODE ***
    # Sample associated with Q(s',a') at kth step
    q_k = [q_table[(s, a)] for (s, a) in q_table.keys() if s == next_state]
    if next_state == mdp.terminal:
        sample = reward
    else:
        sample = reward + mdp.config.gamma*max(q_k)

    # Incorporate new estimate to running avg
    q_kplus1 = (1-alpha)*q_table[(state, action)]+alpha*sample

    # Update the q_table
    q_table[(state, action)] = q_kplus1


def extract_v_table(mdp: tm.TohMdp, q_table: tm.QTable) -> tm.VTable:
    """Extract the value table from the Q-Value table.

    Args:
        mdp: the MDP definition.
        q_table: the Q-Value table to extract values from.

    Returns:
        v_table: tm.VTable
            The extracted value table.
    """
    # *** BEGIN OF YOUR CODE ***
    v_table = {}
    for (s, a) in q_table.keys():  # search all q-table state, action tuples
        if s in v_table.keys():
            if q_table[(s, a)] > v_table[s]:  # update v-table state value if max
                v_table[s] = q_table[(s, a)]
        else:
            v_table[s] = q_table[(s, a)]  # update v-table state value if empty
    return v_table


def choose_next_action(
        mdp: tm.TohMdp, state: tm.TohState, epsilon: float, q_table: tm.QTable,
        epsilon_greedy: Callable[[List[tm.TohAction], float], tm.TohAction]
) -> tm.TohAction:
    """Use the epsilon greedy function to pick the next action.

    You can assume that the passed in state is neither the terminal state nor
    any goal state.

    You can think of the epsilon greedy function passed in having the following
    definition:

    def epsilon_greedy(best_actions, epsilon):
        # selects one of the best actions with probability 1-epsilon,
        # selects a random action with probability epsilon
        ...

    See the concrete definition in QLearningSolver.epsilon_greedy.

    Args:
        mdp: the MDP definition.
        state: the current MDP state.
        epsilon: epsilon value in epsilon greedy.
        q_table: the current Q-value table.
        epsilon_greedy: a function that performs the epsilon

    Returns:
        action: tm.TohAction
            The chosen action.
    """
    # *** BEGIN OF YOUR CODE ***
    best = -10000
    best_actions = []  # initialize best actions list
    for (s, a) in q_table.keys():
        if s == state:  # explore only the current state in the q-table
            if q_table[(s, a)] > best:  # clear the best actions list if Q greater than best
                best_actions = []
            if q_table[(s, a)] >= best:
                best = q_table[(s, a)]  # update best
                best_actions.append(a)  # update best actions
    return epsilon_greedy(best_actions, epsilon)


def custom_epsilon(n_step: int) -> float:
    """Calculates the epsilon value for the nth Q learning step.

    Define a function for epsilon based on `n_step`.

    Args:
        n_step: the nth step for which the epsilon value will be used.

    Returns:
        epsilon: float
            epsilon value when choosing the nth step.
    """
    # *** BEGIN OF YOUR CODE ***
    epsilon = 2**(-1/n_step)  # probability decreases exponentially as steps increase
    return max(epsilon, 0.2)  # set lower bound to avoid sub-optimal policy


def custom_alpha(n_step: int) -> float:
    """Calculates the alpha value for the nth Q learning step.

    Define a function for alpha based on `n_step`.

    Args:
        n_step: the nth update for which the alpha value will be used.

    Returns:
        alpha: float
            alpha value when performing the nth Q update.
    """
    # *** BEGIN OF YOUR CODE ***
    return 2**(-1/n_step)  # decreasing learning rate gives converging average
