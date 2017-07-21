from __future__ import division
import Hand
import Deck
import Constants
import numpy as np
import Game
import math


def dot_product(a, b):
    a = a.tolist() if len(a.shape) != 1 else [a.tolist()]
    b = b.tolist() if len(b.shape) != 1 else [b.tolist()]

    #lxm and mxn matrix
    l = len(a)
    m = len(b)
    n = len(b[0])

    c = [[0]*n for _ in range(l)]

    for i in range(l):
        for j in range(n):
            for k in range(m):
                    c[i][j] += a[i][k]*b[k][j]

    if len(a) == 1 or len(b) == 1:
        return np.array(c).flatten()
    return np.array(c)


def transpose(a):
    a = a.tolist()

    n = len(a)
    m = len(a[0])

    b = [[0]*n for _ in range(m)]

    for i in range(n):
        for j in range(m):
            b[j][i] = a[i][j]
    return np.array(b)

def relu(a, b):
    a = a.tolist() if len(a.shape) != 1 else [a.tolist()]
    b = b.tolist() if len(b.shape) != 1 else [b.tolist()]

    for i in range(len(a)):
        for j in range(len(a[0])):
            if b[i][j] < 0:
                a[i][j] = 0
    if len(a) == 1:
        return np.array(a).flatten()
    return np.array(a)


def softmax(a):
    a = a.tolist() if len(a.shape) != 1 else [a.tolist()]
    c = [0] * len(a[0])
    sum = 0
    for i in range(len(a)):
        sum += math.exp(a[0][i])

    for i in range(len(a)):
        c[i] = math.exp(a[0][i]) / sum

    if len(a) == 1:
        return np.array(c).flatten()
    return np.array(c)

def softmax_cpu(x):
    return np.exp(x) / np.exp(x).sum(axis=0, keepdims=True)

def forward_policy(x):
    # W.T*x
    h = dot_product(x, model['W1'])

    h = relu(h,h)

    logp = dot_product(h, model['W2'])

    p = softmax_cpu(np.clip(logp, -100, 80))

    if np.isnan(p).any():
        print p

    return p,h


def forward_policy_P2(x):
    # W.T*x
    h = dot_product(x, model2['W1'])
    # ReLU activation function

    h = relu(h, h)

    logp = dot_product(h, model2['W2'])

    p = softmax(np.clip(logp, -100, 80))

    if np.isnan(p).any():
        print p

    return p,h


def backward_policy(eph, epdlogp, epx):
    eph_T = transpose(eph)
    dW2 = dot_product(eph_T, epdlogp)
    W2_T = transpose(model['W2'])
    dh = dot_product(epdlogp, W2_T)
    dh = relu(dh, eph)

    epx_T = transpose(epx)
    dW1 = dot_product(epx_T, dh)

    return {'W1': dW1, 'W2': dW2}


# Pick highest probability action, with a certain probability of picking a different choice
def pick_action(action_prob):
    r = np.random.uniform()
    total = 0

    if prob_i % 10 == 0:
        print(prob_i)
        print(action_prob)
    for i, p in enumerate(action_prob):
        total += p
        if r <= total:
            return i

    # action_prob should always sum to 1. This is in case of small rounding error
    return i


# Compute discount rewards. Give more recent rewards more weight.
def discount_rewards(rewards, discount_factor=0.98):
    discounted_r = np.zeros_like(rewards).astype(np.float32)
    running_add = 0
    for i in reversed(xrange(0, len(rewards))):

        # Final state of round
        if rewards[i] != 0:
            running_add = 0

        running_add = running_add * discount_factor + rewards[i]
        discounted_r[i] = running_add
    return discounted_r


# ------------------------------------------------------------------
game = Game.Game()

num_inputs = np.int32(52 * 3)
num_outputs = np.int32(3)  # CALL/CHECK, RAISE, FOLD
num_hiddens = [np.int32(1024)]  # Each value represents number of nodes per layer
NUM_EPISODES = 1000
LEARNING_RATE = 1e-3
GAMMA = 0.99  # discount factor for reward
DECAY_RATE = 0.99  # decay factor for RMSProp leaky sum of grad^2
actions = []
reward_count = []
model = {}
model['W1'] = 0.1 * np.random.randn(num_inputs, num_hiddens[0]).astype(np.float32) / np.sqrt(num_inputs)
model['W2'] = 0.1 * np.random.randn(num_hiddens[0], num_outputs).astype(np.float32) / np.sqrt(num_hiddens[0])
model2 = {'W1': np.copy(model['W1']), 'W2': np.copy(model['W2'])}
grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}
prob_i = 0


def next_round():
    game.next_round()


def update_learning_params(xs, hs, dlogps, rewards, action_raise=False):
    action_prob, h = forward_policy(game.get_input())

    action = pick_action(action_prob)

    # Raise only once. If the chosen action is to raise again, check instead.
    if Constants.ACTIONS[action] == "RAISE" and action_raise:
        action = 0

    hs.append(h)

    # https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
    dlogsoftmax = np.reshape(action_prob, [1, 3])
    dlogsoftmax[0, action] -= 1
    dlogps.append(dlogsoftmax)

    # Observation/input
    # xs.append(game._bot1.get_hand().flatten().astype(np.float32))
    xs.append(game.get_input())

    return action


def opponent_action(action_raise=False):
    action_prob, _ = forward_policy_P2(game.get_input_P2())
    action = np.argmax(action_prob)

    # Raise only once. If the chosen action is to raise again, check instead.
    if Constants.ACTIONS[action] == "RAISE" and action_raise:
        return 0

    return action


# Allow only one Raise per state. Return True if action for either player was FOLD
def handle_action(action, rewards):
    if Constants.ACTIONS[action] == "FOLD":
        rewards.append(-1 * game.get_num_bets())
        return True
    elif Constants.ACTIONS[action] == "RAISE":
        # ALREADY RAISED, why making another action?
        # action = update_learning_params(xs, hs, dlogps, rewards, action_raise=True)
        # rewards.append(0)
        bot2_action = opponent_action(action_raise=False)
        # return handle_action(action, rewards, bot_can_raise=False)
    else:
        bot2_action = opponent_action(action_raise=True)

    if Constants.ACTIONS[bot2_action] == "FOLD":
        rewards.append(1 * game.get_num_bets())
        return True
    elif Constants.ACTIONS[bot2_action] == "RAISE":
        action = update_learning_params(xs, hs, dlogps, rewards, action_raise=True)
        rewards.append(0)
        if Constants.ACTIONS[action] == "FOLD":
            rewards.append(-1 * game.get_num_bets())
            return True

    if Constants.ACTIONS[bot2_action] == "RAISE" or Constants.ACTIONS[action] == "RAISE":
        game.add_bet()

    # Otherwise, both players have checked
    rewards.append(0)
    return False


import time

start = time.time()
xs, hs, dlogps, rewards = [], [], [], []
for i in range(NUM_EPISODES):
    prob_i = i
    game.new_game()

    for _ in range(3):

        action = update_learning_params(xs, hs, dlogps, rewards)
        is_fold = handle_action(action, rewards)

        if is_fold:
            break

        next_round()

    if not is_fold:
        action = update_learning_params(xs, hs, dlogps, rewards)
        is_fold = handle_action(action, rewards)

    if not is_fold:
        # Remove the last reward. The length of rewards must be equal to the number of actions
        rewards.pop()
        # EVALUATE WINNER
        reward = game.evaluate_winner(game.get_p1_hand(), game.get_p2_hand())

        rewards.append(reward * game.get_num_bets())  # +1 / -1 depending on who wins. 0 for tie

    # print(rewards)

    # DEBUG
    reward_count.append(rewards[-1])

    if i > 0 and i % 10 == 0:
        rewards = discount_rewards(rewards)

        # standardize the rewards
        # (Special Case) Don't standardize if someone folds at the start (preflop)
        if len(rewards) != 1:
            rewards -= np.mean(rewards).astype(np.float32)
            rewards /= np.std(rewards).astype(np.float32)

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(rewards)
        epdlogp *= epr

        grad = backward_policy(eph, epdlogp, epx)

        for k in model:
            # accumulate grad over batch
            grad_buffer[k] += grad[k]

        if i % 100 == 0:
            for k, v in model.iteritems():
                g = grad_buffer[k]  # gradient

                # RMSprop: Gradient descent optimization algorithms
                rmsprop_cache[k] = DECAY_RATE * rmsprop_cache[k] + (1 - DECAY_RATE) * g ** 2

                # Update weights to minimize the error
                model[k] -= LEARNING_RATE * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)

                # Reset grad buffer
                grad_buffer[k] = np.zeros_like(v)

                if np.isnan(model['W1']).any():
                    print(1)

        xs, hs, dlogps, rewards = [], [], [], []

    if i > 0 and i % 10 == 0:
        x = np.array(reward_count)
        unique, counts = np.unique(x, return_counts=True)
        values = np.asarray((unique, counts)).T
        earning = 0
        for i in range(values.shape[0]):
            earning += values[i][1] * values[i][0]
        print(earning)
        reward_count = []

    if i > 0 and i % 1000 == 0:
        model2 = {'W1': np.copy(model['W1']), 'W2': np.copy(model['W2'])}

x = np.array(reward_count)
unique, counts = np.unique(x, return_counts=True)

end = time.time()
print(end - start)


values = np.asarray((unique, counts)).T
print(values)
earning = 0
for i in range(11):
    earning += values[i][1] * values[i][0]
loss = 0
for i in range(0, 5):
    loss += values[i][1] * values[i][0]
win = 0
for i in range(6, 11):
    win += values[i][1] * values[i][0]
print(earning)
print(win)
print(loss)
