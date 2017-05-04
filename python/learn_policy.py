# -*- coding: utf-8 -*-
# learn_policy.py
# Katsuki Ohto

import sys, time, datetime

import numpy as np
import tensorflow as tf

from shogi import *
#import model.policy170405 as mdl
#import model.policy170409 as mdl
import model.policy170419 as mdl

N_PHASES = 1
N_TRAIN_BATCHES = 16300
N_RECONSTRUCT_BATCHES = 128
N_TEST_BATCHES = 128

original_learning_rate = 0.0003
learning_rate_decay = 0.999999
test_periods = 2048

#model_file_name = "policy170405"
#model_file_name = "policy170409"
model_file_name = "policy170420"

def learn(data_path, mdl_file, batch_size, mepoch):
    with tf.Graph().as_default():
        x_placeholder = tf.placeholder("float", shape = mdl.IMAGE_INPUT_SHAPE, name = "input")
        y_placeholder = tf.placeholder("float", shape = mdl.IMAGE_MOVE_SUPERVISOR_SHAPE, name = "move")
        learning_rate_placeholder = tf.placeholder("float", shape = (), name = "learning_rate")

        forward_op = [mdl.forward_function(x_placeholder) for ph in range(N_PHASES)] # 順伝播
        normalize_op = [mdl.normalize_function(forward_op[ph]) for ph in range(N_PHASES)] # シグモイド正規化
        loss_op = [mdl.loss_function(normalize_op[ph], y_placeholder) for ph in range(N_PHASES)] # loss計算
        train_op = [mdl.train_function(loss_op[ph], learning_rate_placeholder) for ph in range(N_PHASES)] # training
        test_op = [mdl.accuracy_function(forward_op[ph], y_placeholder) for ph in range(N_PHASES)]  # 一致数計算
        
        # 保存の準備
        saver = tf.train.Saver()
        # Sessionの作成
        sess = tf.Session()
        # 変数の初期化
        sess.run(tf.global_variables_initializer())
        # パラメータ読み込み
        if mdl_file != "none":
            saver.restore(sess, mdl_file)

        learning_rate = original_learning_rate
        
        for e in range(mepoch):
            if e % test_periods == 0:
                # reconstruct phase
                for ph in range(N_PHASES):
                    correct_num = 0
                    for i in range(N_RECONSTRUCT_BATCHES):
                        # データロード
                        index = np.random.randint(N_TRAIN_BATCHES)
                        rec_inputs, rec_moves = mdl.load_npz_input_moves(data_path, index)
                        # 駒打ちなら1, そうでなければ0が立つフラグを用意
                        #rec_drop_flags = np.array([float(rec_moves[i][:, 3:].max()) for i in range(rec_moves.shape[0])])
                        for i in range(0, mdl.BATCH_SIZE, batch_size):
                            correct_num += sess.run(test_op[ph],
                                                    feed_dict={
                                                        x_placeholder: rec_inputs[i: (i + batch_size)],
                                                        y_placeholder: rec_moves[i: (i + batch_size)]})
                    print("[%d samples] train accuracy %g (%d / %d) %s" % (mdl.BATCH_SIZE * e, correct_num / float(mdl.BATCH_SIZE * N_RECONSTRUCT_BATCHES), correct_num, mdl.BATCH_SIZE * N_RECONSTRUCT_BATCHES, datetime.datetime.today()))

                # test phase
                for ph in range(N_PHASES):
                    correct_num = 0
                    for i in range(N_TEST_BATCHES):
                        # データロード
                        index = N_TRAIN_BATCHES + i
                        test_inputs, test_moves = mdl.load_npz_input_moves(data_path, index)
                        # 駒打ちなら1, そうでなければ0が立つフラグを用意
                        #test_drop_flags = np.array([float(test_moves[i][:, 3:].max()) for i in range(test_moves.shape[0])])
                        for i in range(0, mdl.BATCH_SIZE, batch_size):
                            # デバッグ用に出力して確かめる
                            #outputs = sess.run(forward_op[ph],
                            #                   feed_dict = {x_placeholder : test_inputs[i : (i + batch_size)]})
                            #test_mm = mdl.move_function(test_moves[i : (i + batch_size)])
                            #output_mm = mdl.move_function(outputs)
                            #print("teacher output")
                            #for j in range(batch_size):
                            #    print("%s %s" % (test_mm[j], output_mm[j]))
                            correct_num += sess.run(test_op[ph],
                                                    feed_dict = {
                                                        x_placeholder : test_inputs[i : (i + batch_size)],
                                                        y_placeholder : test_moves[i : (i + batch_size)]})
                    print("[%d samples] test  accuracy %g (%d / %d) %s" % (mdl.BATCH_SIZE * e, correct_num / float(mdl.BATCH_SIZE * N_TEST_BATCHES), correct_num, mdl.BATCH_SIZE * N_TEST_BATCHES, datetime.datetime.today()))
                save_path = saver.save(sess, model_file_name)

            # training phase
            for ph in range(N_PHASES):
                # データロード
                index = np.random.randint(N_TRAIN_BATCHES)
                inputs, moves = mdl.load_npz_input_moves(data_path, index)
                for i in range(0, mdl.BATCH_SIZE, batch_size):
                    # トレーニング
                    sess.run(train_op[ph],
                             feed_dict = {
                                 x_placeholder : inputs[i : (i + batch_size)],
                                 y_placeholder : moves[i : (i + batch_size)],
                                 learning_rate_placeholder : learning_rate})
            learning_rate *= learning_rate_decay

if __name__ == "__main__":
    # 設定
    mepoch = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    mdl_file = sys.argv[4]
    data_path = sys.argv[1]

    learn(data_path, mdl_file, batch_size, mepoch)