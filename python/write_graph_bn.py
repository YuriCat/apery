# -*- coding: utf-8 -*-
# write_graph_bn.py
# Katsuki Ohto

import sys, time, datetime

import numpy as np
import tensorflow as tf

from shogi import *
#import model.policy170405 as mdl
#import model.policy170409 as mdl
#import model.policy170419 as mdl
#import model.policy170409_bn as mdl
#import model.policy170419_bn as mdl
#import model.policy170425_bn as mdl
import model.policy170428_bn as mdl

def write(input_checkpoint, output_name):
    g1 = tf.Graph()
    vars = {}
    with g1.as_default():
        with tf.Session() as sess:
            x_placeholder = tf.placeholder("float", shape = mdl.IMAGE_INPUT_SHAPE, name = "input")
            is_training_placeholder = tf.placeholder("bool", shape = (), name = "is_training")
            #forward_op = mdl.forward_function(x_placeholder)
            forward_op = mdl.forward_function(x_placeholder, is_training_placeholder)
            normalize_op = mdl.normalize_function(forward_op)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, input_checkpoint)
            #for v in tf.trainable_variables():
            for v in tf.all_variables():
                print(v.value().name)
                vars[v.value().name] = sess.run(v)
    g2 = tf.Graph()
    consts = {}
    with g2.as_default():
        with tf.Session() as sess:
            for k, v in vars.items():
                consts[k] = tf.constant(v)

            tf.import_graph_def(g1.as_graph_def(), input_map=consts, name="g")
            
            print([n.name for n in g2.as_graph_def().node])
            
            tf.train.write_graph(g2.as_graph_def(), '.', output_name + '.pb', as_text=False)
            tf.train.write_graph(g2.as_graph_def(), '.', output_name + '.txt', as_text=True)
if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage : python3 write_graph_bn.py input output")
        exit()
    # 設定
    input_checkpoint = sys.argv[1]
    output_name = sys.argv[2]

    write(input_checkpoint, output_name)
