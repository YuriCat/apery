# -*- coding: utf-8 -*-
# policy170405.py
# Katsuki Ohto

import os, sys
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())

from shogi import *

BATCH_SIZE = 1024

# 将棋盤画像の諸定義
IMAGE_FILE_NUM = 11
IMAGE_RANK_NUM = 11
IMAGE_SIZE = IMAGE_FILE_NUM * IMAGE_FILE_NUM
IMAGE_INPUT_PLAINS = 107

IMAGE_MOVE_CHANNELS = 10 # from 1, to 9

IMAGE_MOVE_OUTPUTS = IMAGE_MOVE_CHANNELS * IMAGE_SIZE

IMAGE_INPUT_SHAPE = (None, IMAGE_FILE_NUM, IMAGE_RANK_NUM, IMAGE_INPUT_PLAINS)
IMAGE_MOVE_OUTPUT_SHAPE = (None, IMAGE_SIZE, IMAGE_MOVE_CHANNELS)
IMAGE_MOVE_SUPERVISOR_SHAPE = (None, IMAGE_SIZE, IMAGE_MOVE_CHANNELS)

def load_npz_input_moves(path, n):
    file_name = path + str(n) + ".npz"
    data = np.load(file_name)
    ret = (data['input'], data['move'])
    data.close()
    return ret

N_FILTERS = 128
N_LAYERS = 15

# neural network
def forward_function(x_placeholder):
    # モデルを作成する関数
    
    # 初期化関数
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.01, dtype = tf.float32)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape = shape, dtype = tf.float32)
        return tf.Variable(initial)

    # 畳み込み層
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

    # 第1層
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([3, 3, IMAGE_INPUT_PLAINS, N_FILTERS])
        b_conv1 = bias_variable([N_FILTERS])
        h_conv = tf.nn.relu(conv2d(x_placeholder, W_conv1) + b_conv1)
    
    layers = range(0, N_LAYERS - 2)
    W_conv = [weight_variable([3, 3, N_FILTERS, N_FILTERS]) for l in layers]
    b_conv = [bias_variable([N_FILTERS]) for l in layers]

    # 第2層~
    for l in layers:
        with tf.name_scope('conv' + str(l + 2)) as scope:
            h_conv = tf.nn.relu(conv2d(h_conv, W_conv[l]) + b_conv[l] + h_conv)

    # 最終層
    with tf.name_scope('last') as scope:
        W_convl = weight_variable([1, 1, N_FILTERS, IMAGE_MOVE_CHANNELS])
        h_convl = conv2d(h_conv, W_convl)
        h_from = h_convl[:, :, :, :1]
        h_to = h_convl[:, :, :, 1:]
        h_from_liner = tf.reshape(h_from, [-1, IMAGE_SIZE])
        h_to_liner = tf.reshape(h_to, [-1, IMAGE_SIZE * (IMAGE_MOVE_CHANNELS - 1)])
        h_liner = tf.concat([h_from_liner, h_to_liner], -1)

        b_last = bias_variable([IMAGE_SIZE * IMAGE_MOVE_CHANNELS])
        h_last = h_liner + b_last

    return h_last

def normalize_function(y):
    return tf.concat([tf.nn.softmax(y[:, :IMAGE_SIZE]),
                      tf.nn.softmax(y[:, IMAGE_SIZE:])], -1)

def loss_function(y, answer):
    #cross_entropy = -tf.reduce_sum(move * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

    # to loss
    a_to = answer[:, :, 1:]
    a_to_line = tf.reshape(a_to, [-1, (IMAGE_MOVE_CHANNELS - 1) * IMAGE_SIZE])

    # from loss
    a_from = answer[:, :, :1]
    a_from_line = tf.reshape(a_from, [-1, IMAGE_SIZE])

    loss_to = -tf.reduce_sum(a_to_line * tf.log(tf.clip_by_value(y[:, IMAGE_SIZE:], 1e-10, 1.0)))
    loss_from = -tf.reduce_sum(a_from_line * tf.log(tf.clip_by_value(y[:, :IMAGE_SIZE], 1e-10, 1.0)))

    return loss_to + loss_from

def train_function(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy_function(y, answer, drop_flag):
    # 移動先の一致
    a_to = answer[:, :, 1:]
    y_to_line = y[:, IMAGE_SIZE:]
    a_to_line = tf.reshape(a_to, [-1, (IMAGE_MOVE_CHANNELS - 1) * IMAGE_SIZE])
    y_to_amax = tf.argmax(y_to_line, 1)
    a_to_amax = tf.argmax(a_to_line, 1)
    correct_to = tf.to_float(tf.equal(y_to_amax, a_to_amax))

    # 移動元の一致
    a_from = answer[:, :, :1]
    y_from_line = y[:, :IMAGE_SIZE]
    a_from_line = tf.reshape(a_from, [-1, IMAGE_SIZE])
    y_from_amax = tf.argmax(y_from_line, 1)
    a_from_amax = tf.argmax(a_from_line, 1)
    correct_from = tf.to_float(tf.equal(y_from_amax, a_from_amax))

    # 駒打ちの場合は移動元は正解として一致するか計算
    correct = correct_to * tf.maximum(correct_from, drop_flag)

    # 一致数を返す
    return tf.reduce_sum(tf.to_int32(correct))

def move_function(y):
    # forward出力から着手形式に変換
    # Tensor型の演算ではない
    y_to = y[:, :, 1:]
    #print(y_to.shape)

    y_to_line = np.reshape(y_to, [-1, (IMAGE_MOVE_CHANNELS - 1) * IMAGE_SIZE])
    y_to_amax = np.argmax(y_to_line, 1)

    #print(y_to_amax)

    y_from = y[:, :, :1]
    y_from_line = np.reshape(y_from, [-1, IMAGE_SIZE])
    y_from_amax = np.argmax(y_from_line, 1)

    #print(y_from_amax)

    moves = []
    for i in range(len(y_to_amax)):
        psq_to = y_to_amax[i] // (IMAGE_MOVE_CHANNELS - 1)
        channel_to = y_to_amax[i] % (IMAGE_MOVE_CHANNELS - 1)
        psq_from = y_from_amax[i]

        promotion = False

        if channel_to >= 2: # drop
            #print(channel_to - 2)
            sq_from = 121 + channel_to - 2
        else:
            sq_from = make_square((psq_from // IMAGE_RANK_NUM) - 1,
                                  (psq_from % IMAGE_RANK_NUM) - 1)
            if channel_to >= 1:
                promotion = True

        sq_to = make_square((psq_to // IMAGE_RANK_NUM) - 1,
                            (psq_to % IMAGE_RANK_NUM) - 1)
        #print(sq_from, sq_to, promotion)
        moves.append(MinimumMove(sq_from, sq_to, promotion))

    return moves

if __name__ == '__main__':

    ys = np.zeros((2, IMAGE_SIZE, IMAGE_MOVE_CHANNELS))

    #(15,20+)
    ys[0][(make_file(15) + 1) * IMAGE_RANK_NUM + make_rank(15) + 1][0] = 1
    ys[0][(make_file(20) + 1) * IMAGE_RANK_NUM + make_rank(20) + 1][2] = 1

    #(S*,50)
    ys[1][(make_file(50) + 1) * IMAGE_RANK_NUM + make_rank(50) + 1][6] = 1

    moves = move_function(ys)
    for move in moves:
        print(move)



