from __future__ import division
import Constants
import numpy as np
import Game


def softmax_cpu(x):
    return np.exp(x) / np.exp(x).sum(axis=0, keepdims=True)


def forward_policy(x):
    # W.T*x
    h = np.dot(x, model['W1'])
    # ReLU activation function

    h[h < 0] = 0
    logp = np.dot(h, model['W2'])

    # Add ceiling and floor values to avoid overflow
    p = softmax_cpu(np.clip(logp, -100, 80))

    return p, h


def forward_policy_P2(x):
    # W.T*x
    h = np.dot(x, model2['W1'])
    # ReLU activation function

    h[h < 0] = 0
    logp = np.dot(h, model2['W2'])

    print(h.shape)
    print(logp.shape)
    # Add ceiling and floor values to avoid overflow
    p = softmax_cpu(np.clip(logp, -100, 80))

    return p, h


def backward_policy(eph, epdlogp, epx):
    dW2 = eph.T.dot(epdlogp)

    dh = epdlogp.dot(model['W2'].T)
    dh[eph <= 0] = 0

    dW1 = epx.T.dot(dh)

    return {'W1': dW1, 'W2': dW2}


# Pick highest probability action, with a certain probability of picking a different choice
def pick_action(action_prob):
    r = np.random.uniform()
    total = 0

    for i, p in enumerate(action_prob):
        total += p
        if r <= total:
            return i

    # action_prob should always sum to 1. This is in case of small rounding error
    return i


# Compute discount rewards. Give more recent rewards more weight.
def discount_rewards(rewards, discount_factor=Constants.GAMMA):
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

num_inputs = np.int32(Constants.HAND_INPUT_SIZE)
num_outputs = np.int32(Constants.NUM_OUTPUTS)  # CALL/CHECK, RAISE, FOLD
num_hiddens = [np.int32(Constants.NUM_NODE_HIDDEN)]  # Each value represents number of nodes per layer
actions = []
reward_count = []
model = {}

if Constants.RANDOM_WEIGHT_INIT:
    model['W1'] = 0.1 * np.random.randn(num_inputs, num_hiddens[0]).astype(np.float32)
    model['W2'] = 0.1 * np.random.randn(num_hiddens[0], num_outputs).astype(np.float32)
    # Uncomment to save weights
    #model['W1'].tofile("W1.txt")
    #model['W2'].tofile("W2.txt")
else:
    model['W1'] = np.fromfile("W1.txt", dtype=np.float32).reshape((num_inputs, num_hiddens[0]))
    model['W2'] = np.fromfile("W2.txt", dtype=np.float32).reshape((num_hiddens[0], num_outputs))

model2 = {'W1': np.copy(model['W1']), 'W2': np.copy(model['W2'])}
grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}
prob_i = 0


def next_round():
    game.next_round()


def update_learning_params(xs, hs, dlogps, action_raise=False):
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
        # bot2_action = game.bot_decision(can_raise=False)
        bot2_action = opponent_action(action_raise=False)
    else:
        # bot2_action = game.bot_decision(can_raise=True)
        bot2_action = opponent_action(action_raise=True)

    if Constants.ACTIONS[bot2_action] == "FOLD":
        rewards.append(1 * game.get_num_bets())
        return True
    elif Constants.ACTIONS[bot2_action] == "RAISE":
        action = update_learning_params(xs, hs, dlogps, action_raise=True)
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
for i in range(Constants.NUM_OF_EPS):
    prob_i = i
    game.new_game()

    for _ in range(3):

        action = update_learning_params(xs, hs, dlogps)
        is_fold = handle_action(action, rewards)

        if is_fold:
            break

        next_round()

    if not is_fold:
        action = update_learning_params(xs, hs, dlogps)
        is_fold = handle_action(action, rewards)

    if not is_fold:
        # Remove the last reward. The length of rewards must be equal to the number of actions
        rewards.pop()
        # EVALUATE WINNER
        reward = game.evaluate_winner(game.get_p1_hand(), game.get_p2_hand())

        rewards.append(reward * game.get_num_bets())  # +1 / -1 depending on who wins. 0 for tie

    # DEBUG
    reward_count.append(rewards[-1])

    if i % 100 == 0:
        rewards = discount_rewards(rewards)

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(rewards)
        epdlogp *= epr

        grad = backward_policy(eph, epdlogp, epx)

        for k in model:
            # accumulate grad over batch
            grad_buffer[k] += grad[k]

        if i % 200 == 0:
            for k, v in model.iteritems():
                g = grad_buffer[k]  # gradient

                # RMSprop: Gradient descent optimization algorithms
                rmsprop_cache[k] = Constants.DECAY_RATE * rmsprop_cache[k] + (1 - Constants.DECAY_RATE) * g ** 2

                # Update weights to minimize the error
                model[k] -= Constants.LEARNING_RATE * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)

                # Reset grad buffer
                grad_buffer[k] = np.zeros_like(v)

        xs, hs, dlogps, rewards = [], [], [], []

    if i > 0 and i % 200 == 0:
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
