# -*- coding: utf-8 -*-
# shogi_image.py
# Katsuki Ohto

import numpy as np

import shogi

BATCH = 256

# 将棋盤画像の諸定義
IMAGE_ROWS = 11
IMAGE_COLS = 11
IMAGE_SIZE = IMAGE_ROWS * IMAGE_COLS
IMAGE_INPUT_PLAINS = 103

IMAGE_MOVE_CHANNELS = 10 # from 1, to 9

IMAGE_MOVE_OUTPUTS = IMAGE_MOVE_CHANNELS * IMAGE_SIZE

def load_npz_input_moves(path, n):
    input_file_name = path + "input" + str(n) + ".npz"
    move_file_name = path + "move" + str(n) + ".npz"
    inputz = np.load(input_file_name)
    movez = np.load(move_file_name)
    ret = (inputz['arr_0'], movez['arr_0'])
    inputz.close()
    movez.close()
    return ret