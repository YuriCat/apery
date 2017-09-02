# -*- coding: utf-8 -*-
# learn_pv_hcpe_bn.py
# Katsuki Ohto

import sys, time, datetime

import numpy as np
import tensorflow as tf

from shogi import *

import model.pv170518_bn as mdl

N_PHASES = 1

original_learning_rate = 0.0001 # 初期の学習率
learning_rate_decay = 0.999999999 # 学習率の減衰率 (1局面あたり)
#test_periods = 1000000 # テストを行う局面頻度
test_time_periods = datetime.timedelta(seconds=60 * 30)# テストを行う時間頻度
reconstruct_num = 2 ** 17
test_num = 2 ** 17

model_file_name = "pv170518_bn"

def learn(mdl_file, train_data_path, test_data_path, data_batch_size, batch_size, mepoch):
    with tf.Graph().as_default():
        # プレースホルダー定義
        x_placeholder = tf.placeholder(tf.float32, shape=mdl.IMAGE_INPUT_SHAPE, name="input")
        move_placeholder = tf.placeholder(tf.float32, shape=mdl.IMAGE_MOVE_OUTPUT_SHAPE, name="move")
        move_label_placeholder = tf.placeholder(tf.int64, shape=mdl.IMAGE_MOVE_LABEL_SHAPE, name="move_label")
        value_placeholder = tf.placeholder(tf.float32, shape=(None), name="value")
        is_training_placeholder = tf.placeholder(tf.bool, shape=(), name="is_training")
        learning_rate_placeholder = tf.placeholder(tf.float32, shape=(), name="learning_rate")

        # オペレーション定義
        forward_op = [mdl.forward_function(x_placeholder, is_training_placeholder) for ph in range(N_PHASES)] # 順伝播
        normalize_op = [mdl.normalize_function(forward_op[ph]) for ph in range(N_PHASES)] # 正規化
        loss_op = [mdl.loss_function(normalize_op[ph], move_placeholder, value_placeholder) for ph in range(N_PHASES)] # loss計算
        train_op = [mdl.train_function(loss_op[ph], learning_rate_placeholder) for ph in range(N_PHASES)] # training
        test_op = [mdl.test_function(normalize_op[ph], move_placeholder, value_placeholder) for ph in range(N_PHASES)]  # テスト計算

        move_loss_op = [mdl.move_loss_function(normalize_op[ph], move_placeholder) for ph in range(N_PHASES)] # loss計算 (moveのみ)
        move_train_op = [mdl.train_function(move_loss_op[ph], learning_rate_placeholder) for ph in range(N_PHASES)] # training (moveのみ)

        value_loss_op = [mdl.value_loss_function(normalize_op[ph], value_placeholder) for ph in range(N_PHASES)]  # loss計算 (valueのみ)
        value_train_op = [mdl.train_function(value_loss_op[ph], learning_rate_placeholder) for ph in range(N_PHASES)]  # training (valueのみ)
        
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
        policy_learned_num, value_learned_num = 0, 0
        tstart = datetime.datetime(2017, 5, 3, 16, 30)
        last_test_timing = -10000000000
        
        for e in range(mepoch):
            if datetime.datetime.today() - tstart >= test_time_periods: # 最初は必ずテストを呼ぶ
                # 時間計測初期化
                tstart = datetime.datetime.today()
            #if policy_learned_num - last_test_timing >= test_periods:
                # reconstruct phase
                print("[%d policy samples, %d value samples] %s" % (policy_learned_num, value_learned_num,
                                                                    datetime.datetime.today()))
                for ph in range(N_PHASES):
                    test_result_sum = np.zeros(4)
                    for j in range(0, reconstruct_num, data_batch_size):
                        inputs, moves, move_labels, values = mdl.load_inputs_moves_labels_values(train_data_path, data_batch_size)
                        """print(inputs.shape, moves.shape, move_labels.shape, values.shape)
                        print(type(move_labels[0][0]))
                        print(move_labels)
                        print(moves)"""
                        """print(sess.run(train_op[ph], feed_dict={
                            x_placeholder: inputs,
                            move_placeholder: moves,
                            value_placeholder: values,
                            is_training_placeholder: True,
                            learning_rate_placeholder: learning_rate}))
                        print(sess.run(value_loss_op[ph], feed_dict={
                            x_placeholder: inputs,
                            value_placeholder: values,
                            is_training_placeholder: True}))"""
                        """print(sess.run(test_op[ph], feed_dict={
                            x_placeholder: inputs,
                            move_placeholder: moves,
                            #move_label_placeholder: move_labels,
                            value_placeholder: values,
                            is_training_placeholder: False}))
                        i = 0
                        print(inputs[i:(i + batch_size)].shape)
                        print(moves[i:(i + batch_size)].shape)
                        print(move_labels[i:(i + batch_size)].shape)
                        print(values[i:(i + batch_size)].shape)
                        print(mdl.IMAGE_INPUT_SHAPE)
                        print(mdl.IMAGE_MOVE_OUTPUT_SHAPE)
                        print(mdl.IMAGE_MOVE_LABEL_SHAPE)"""
                        for i in range(0, data_batch_size, batch_size):
                            test_result_sum += sess.run(test_op[ph],
                                                        feed_dict={
                                                            x_placeholder: inputs[i:(i + batch_size)],
                                                            move_placeholder: moves[i:(i + batch_size)],
                                                            #move_label_placeholder: move_labels[i:(i + batch_size)],
                                                            value_placeholder: values[i:(i + batch_size)],
                                                            is_training_placeholder: False})
                    test_result_sum /= reconstruct_num
                    print("train - move: loss %g acc %g value: loss %g mae %g  (/%d)" % (test_result_sum[0], test_result_sum[1],
                                                                                         test_result_sum[2], test_result_sum[3],
                                                                                         reconstruct_num))

                # test phase
                print("[%d policy samples, %d value samples] %s" % (policy_learned_num, value_learned_num,
                                                                    datetime.datetime.today()))
                for ph in range(N_PHASES):
                    test_result_sum = np.zeros(4)
                    for j in range(0, test_num, data_batch_size):
                        inputs, moves, move_labels, values = mdl.load_inputs_moves_labels_values(test_data_path, data_batch_size)
                        for i in range(0, data_batch_size, batch_size):
                            test_result_sum += sess.run(test_op[ph],
                                                        feed_dict={
                                                            x_placeholder: inputs[i:(i + batch_size)],
                                                            move_placeholder: moves[i:(i + batch_size)],
                                                            #move_label_placeholder: move_labels[i:(i + batch_size)],
                                                            value_placeholder: values[i:(i + batch_size)],
                                                            is_training_placeholder: False})
                    test_result_sum /= test_num
                    print("test  - move: loss %g acc %g value: loss %g mae %g  (/%d)" % (test_result_sum[0], test_result_sum[1],
                                                                                         test_result_sum[2], test_result_sum[3],
                                                                                         test_num))
                # モデル保存
                save_path = saver.save(sess, model_file_name)
                last_test_timing = policy_learned_num

            # trainig phase (policy - value 同時)
            for ph in range(N_PHASES):
                inputs, moves, move_labels, values = mdl.load_inputs_moves_labels_values(train_data_path, data_batch_size)
                for i in range(0, data_batch_size, batch_size):
                    sess.run(train_op[ph],
                             feed_dict = {
                                 x_placeholder: inputs[i:(i + batch_size)],
                                 move_placeholder: moves[i:(i + batch_size)],
                                 value_placeholder: values[i:(i + batch_size)],
                                 is_training_placeholder: True,
                                 learning_rate_placeholder: learning_rate})
                # 学習局面数更新
                policy_learned_num += data_batch_size
                value_learned_num += data_batch_size

            learning_rate *= learning_rate_decay ** data_batch_size

if __name__ == "__main__":
    # 設定
    data_batch_size = int(sys.argv[1]) # データ取得のバッチサイズ
    batch_size = int(sys.argv[2]) # 学習のバッチサイズ
    train_data_path = sys.argv[3]
    test_data_path = sys.argv[4]
    mdl_file = sys.argv[5]
    mepoch = 1000000000

    learn(mdl_file, train_data_path, test_data_path, data_batch_size, batch_size, mepoch)