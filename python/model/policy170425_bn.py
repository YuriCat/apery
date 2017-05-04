# -*- coding: utf-8 -*-
# policy170425_bn.py
# Katsuki Ohto

import os, sys
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())

from shogi import *
from model.nnparts import *

BATCH_SIZE = 1024

# 将棋盤画像の諸定義
IMAGE_FILE_NUM = 11
IMAGE_RANK_NUM = 11
IMAGE_SIZE = IMAGE_FILE_NUM * IMAGE_FILE_NUM
IMAGE_INPUT_PLAINS = 107

IMAGE_MOVE_FROM_PLAINS = 1
IMAGE_MOVE_DROP_SIZE = 7
IMAGE_MOVE_FROM_SIZE = IMAGE_MOVE_FROM_PLAINS * IMAGE_SIZE + IMAGE_MOVE_DROP_SIZE

IMAGE_MOVE_TO_PLAINS = 14
IMAGE_MOVE_TO_SIZE = IMAGE_MOVE_TO_PLAINS * IMAGE_SIZE

IMAGE_MOVE_OUTPUTS = IMAGE_MOVE_FROM_SIZE + IMAGE_MOVE_TO_SIZE

IMAGE_INPUT_SHAPE = (None, IMAGE_FILE_NUM, IMAGE_RANK_NUM, IMAGE_INPUT_PLAINS)
IMAGE_MOVE_OUTPUT_SHAPE = (None, IMAGE_MOVE_OUTPUTS)
IMAGE_MOVE_SUPERVISOR_SHAPE = (None, IMAGE_MOVE_OUTPUTS)

def load_npz_input_moves(path, n):
    file_name = path + str(n) + ".npz"
    data = np.load(file_name)
    input = data['input']
    move = data['move']
    data.close()
    move_array = np.zeros((BATCH_SIZE, IMAGE_MOVE_OUTPUTS), dtype = np.float32)
    for i in range(BATCH_SIZE):
        move_array[i][int(move[i][0])] = 1
        move_array[i][int(move[i][1])] = 1
    return input, move_array

N_FILTERS = 128
N_LAYERS = 13

# neural network
def forward_function(x_placeholder, is_training_placeholder):
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
        W_conv1_1x1 = weight_variable([1, 1, IMAGE_INPUT_PLAINS, N_FILTERS])
        W_conv1_3x3 = weight_variable([3, 3, IMAGE_INPUT_PLAINS, N_FILTERS])
        W_conv1_5x5 = weight_variable([5, 5, IMAGE_INPUT_PLAINS, N_FILTERS])
        W_conv1_7x7 = weight_variable([7, 7, IMAGE_INPUT_PLAINS, N_FILTERS])
        W_conv1_9x9 = weight_variable([9, 9, IMAGE_INPUT_PLAINS, N_FILTERS])
        W_conv1_11x11 = weight_variable([11, 11, IMAGE_INPUT_PLAINS, N_FILTERS])
        b_conv1 = bias_variable([N_FILTERS])
        h_conv_1x1 = conv2d(x_placeholder, W_conv1_1x1)
        h_conv_3x3 = conv2d(x_placeholder, W_conv1_3x3)
        h_conv_5x5 = conv2d(x_placeholder, W_conv1_5x5)
        h_conv_7x7 = conv2d(x_placeholder, W_conv1_7x7)
        h_conv_9x9 = conv2d(x_placeholder, W_conv1_9x9)
        h_conv_11x11 = conv2d(x_placeholder, W_conv1_11x11)
        h_conv = batch_norm(tf.nn.relu(h_conv_1x1 + h_conv_3x3 + h_conv_5x5 + h_conv_7x7 + h_conv_9x9 + h_conv_11x11 + b_conv1),
                            N_FILTERS, is_training_placeholder)
    
    layers = range(0, N_LAYERS - 2)
    W_conv = [weight_variable([3, 3, N_FILTERS, N_FILTERS]) for l in layers]
    b_conv = [bias_variable([N_FILTERS]) for l in layers]

    # 第2層~
    for l in layers:
        with tf.name_scope('conv' + str(l + 2)) as scope:
            h_conv = batch_norm(tf.nn.relu(conv2d(h_conv, W_conv[l]) + b_conv[l] + h_conv),
                                N_FILTERS, is_training_placeholder)

    # move最終層
    with tf.name_scope('move_last') as scope:
        # from
        W_conv_all = weight_variable([1, 1, N_FILTERS, IMAGE_MOVE_FROM_PLAINS + IMAGE_MOVE_TO_PLAINS + IMAGE_MOVE_DROP_SIZE])
        h_conv_all = conv2d(h_conv, W_conv_all)
        h_from_move = tf.reshape(h_conv_all[:, :, :, :IMAGE_MOVE_FROM_PLAINS], [-1, IMAGE_SIZE * IMAGE_MOVE_FROM_PLAINS])
        h_to = tf.reshape(h_conv_all[:, :, :, (IMAGE_MOVE_FROM_PLAINS + IMAGE_MOVE_DROP_SIZE):], [-1, IMAGE_MOVE_TO_SIZE])
        # from(drop)
        h_conv_drop = h_conv_all[:, :, :, IMAGE_MOVE_FROM_PLAINS : (IMAGE_MOVE_FROM_PLAINS + IMAGE_MOVE_DROP_SIZE)]
        h_drop = tf.reshape(h_conv_drop, [-1, IMAGE_SIZE, IMAGE_MOVE_DROP_SIZE])
        h_drop = tf.reduce_sum(h_drop, -2)
        # all
        h_all = tf.concat([h_from_move, h_drop, h_to], -1)
        b_all = bias_variable([IMAGE_MOVE_OUTPUTS])
        o_move = h_all + b_all
    return o_move

def normalize_function(y):
    with tf.name_scope('normalize') as scope:
        prob = tf.concat([tf.nn.softmax(y[:, :IMAGE_MOVE_FROM_SIZE]),
                          tf.nn.softmax(y[:, IMAGE_MOVE_FROM_SIZE:])], -1)
    return prob

def loss_function(y, a):
    loss_to = -tf.reduce_sum(a[:, IMAGE_MOVE_FROM_SIZE:] * tf.log(tf.clip_by_value(y[:, IMAGE_MOVE_FROM_SIZE:], 1e-10, 1.0)))
    loss_from = -tf.reduce_sum(a[:, :IMAGE_MOVE_FROM_SIZE] * tf.log(tf.clip_by_value(y[:, :IMAGE_MOVE_FROM_SIZE], 1e-10, 1.0)))

    return loss_to + loss_from

def train_function(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy_function(y, a):
    # 移動先の一致
    y_to_amax = tf.argmax(y[:, IMAGE_MOVE_FROM_SIZE:], 1)
    a_to_amax = tf.argmax(a[:, IMAGE_MOVE_FROM_SIZE:], 1)
    correct_to = tf.to_float(tf.equal(y_to_amax, a_to_amax))

    # 移動元の一致
    y_from_amax = tf.argmax(y[:, :IMAGE_MOVE_FROM_SIZE], 1)
    a_from_amax = tf.argmax(a[:, :IMAGE_MOVE_FROM_SIZE], 1)
    correct_from = tf.to_float(tf.equal(y_from_amax, a_from_amax))

    correct = correct_to * correct_from

    # 一致数を返す
    return tf.reduce_sum(tf.to_int32(correct))

def move_function(y):
    # forward出力から着手形式に変換
    # Tensor型の演算ではない
    y_to_amax = np.argmax(y[:, IMAGE_MOVE_FROM_SIZE:], 1)
    y_from_amax = np.argmax(y[:, :IMAGE_MOVE_FROM_SIZE], 1)

    moves = []
    for i in range(len(y_to_amax)):
        psq_to = y_to_amax[i] // IMAGE_MOVE_TO_PLAINS
        plane_to = y_to_amax[i] % IMAGE_MOVE_TO_PLAINS
        psq_from = y_from_amax[i]

        if plane_to >= 1:
            promotion = True
        else:
            promotion = False

        if psq_from >= IMAGE_SIZE: # drop
            sq_from = psq_from
        else:
            sq_from = make_square((psq_from // IMAGE_RANK_NUM) - 1,
                                  (psq_from % IMAGE_RANK_NUM) - 1)

        sq_to = make_square((psq_to // IMAGE_RANK_NUM) - 1,
                            (psq_to % IMAGE_RANK_NUM) - 1)
        #print(sq_from, sq_to, promotion)
        moves.append(MinimumMove(sq_from, sq_to, promotion))

    return moves

if __name__ == '__main__':

    ys = np.zeros((2, IMAGE_MOVE_OUTPUTS))

    #(15,20+)
    ys[0][(make_file(15) + 1) * IMAGE_RANK_NUM + make_rank(15) + 1] = 1
    ys[0][IMAGE_MOVE_FROM_SIZE + ((make_file(20) + 1) * IMAGE_RANK_NUM + make_rank(20) + 1) * 2 + 1] = 1

    #(S*,50)
    ys[1][IMAGE_SIZE + 3] = 1
    ys[1][IMAGE_MOVE_FROM_SIZE + ((make_file(50) + 1) * IMAGE_RANK_NUM + make_rank(50) + 1) * 2] = 1

    moves = move_function(ys)
    for move in moves:
        print(move)



