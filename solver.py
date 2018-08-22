# coding:utf-8

import tensorflow as tf
from tensorflow.contrib import layers as tflayers
import queue
from game import GameCore
import numpy as np


def model():
    x = tf.placeholder(tf.float32, [None, 16])
    y = tf.placeholder(tf.float32, [None, 4])
    h_1 = tflayers.fully_connected(x, 256, tf.nn.leaky_relu)
    h_2 = tflayers.fully_connected(h_1, 64, tf.nn.leaky_relu)
    out = tflayers.fully_connected(h_2, 4)
    loss = tf.losses.mean_squared_error(y, out)
    return x, y, out, loss


def normalize(x):
    y = x.copy().reshape(16)
    y[np.where(y == 0)] = 1
    return np.log2(y) / 18.0 + 1.0 / 18.0


if __name__ == '__main__':
    batch_size = 256
    learning_rate = 1e-6
    discount = 0.9
    epsilon = 1
    num_explore = 100000
    data_size = 30000
    epochs = 50000000
    max_score = 0
    m_x, m_y, m_out, m_loss = model()
    op = tf.train.AdamOptimizer(learning_rate).minimize(m_loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    data_set = queue.deque()
    game = GameCore()
    game.new_block()
    for step in range(epochs):
        curr_board = normalize(game.board)
        curr_score = game.score
        epsilon = max(1 - max(step - data_size, 0) / num_explore, 0.01)
        if np.random.rand() < epsilon:
            action = np.random.permutation(4)
        else:
            v_out = sess.run(m_out, feed_dict={m_x: curr_board.reshape(1, 16)})
            action = [i[0] for i in sorted(enumerate(v_out[0]), key=lambda x: x[1], reverse=True)]
        index = game.move(action)
        next_board = normalize(game.board)
        score = np.log2(max(game.score - curr_score, 1))
        reward = np.zeros(4)
        if index >= 0:
            reward[index] = score
        else:
            if step > data_size:
                print("step=%8d, max_score=%d" % (step, max_score))
                game.show()
            reward[index] = -1000.
            max_score = max(max_score, game.score)
            game.__init__()
        game.new_block()
        data_set.append((curr_board, reward, next_board))
        if len(data_set) > data_size:
            data_set.popleft()
        if step > data_size and step % batch_size == 0:
            samples = [data_set[i] for i in np.random.choice(len(data_set), batch_size)]
            curr_boards = np.array([x[0] for x in samples])
            rewards = np.array([x[1] for x in samples])
            next_boards = np.array([x[2] for x in samples])
            v_out = sess.run(m_out, feed_dict={m_x: next_boards})
            v_y = np.add(rewards, discount * np.max(v_out, axis=1).reshape(v_out.shape[0], 1))
            sess.run(op, feed_dict={m_x: curr_boards, m_y: v_y})
        if step % 100000 == 0:
            saver.save(sess, "./models/model.ckpt", global_step=step)
