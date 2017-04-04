# -*- coding: utf-8 -*-
# image_converter.py
# Katsuki Ohto

# 文字列の画像を学習時のnumpy形式に保存

import sys

import numpy as np

BATCH = 256

# データ読み込み & 変換部
IMAGE_ROWS = 11
IMAGE_COLS = 11
IMAGE_SIZE = IMAGE_ROWS * IMAGE_COLS
IMAGE_INPUT_PLAINS = 103

IMAGE_MOVE_CHANNELS = 10 # from 1, to 9

def convert_board(file_path, size):
    # sizeの数が入った画像データを読む
    f = open(file_path)
    images = np.empty((size, IMAGE_ROWS, IMAGE_COLS, IMAGE_PLAINS),
                      dtype = np.float32)
    moves = np.zeros((size, IMAGE_SIZE, IMAGE_MOVE_CHANNELS), dtype = np.float32)
    
    for i, line in enumerate(f):
        #print(i)
        v = line.split(' ')
        # move の処理
        sq_from = int(v[0])
        sq_to = int(v[1])
        promote = int(v[2])
        
        # 移動元
        if sq_from >= IMAGE_SIZE:
            # 駒打ち
            piece = sq_from - IMAGE_SIZE
            moves[i][sq_to][2 + piece] = 1
        else:
            moves[i][sq_from][0] = 1
            moves[i][sq_to][promote] = 1
        
        # input の処理
        for x in xrange(IMAGE_ROWS):
            for y in xrange(IMAGE_COLS):
                binary_str = v[3 + x * IMAGE_COLS + y]
                for p in xrange(IMAGE_INPUT_INPUT_PLAINS):
                    # bitsetを逆順にする
                    images[i][x][y][IMAGE_INPUT_PLAINS - 1 - p] = int(binary_str[p])
    f.close()
    return images, moves

def convert_to_np(ipath, opath, n):
    file_name = ipath + str(n) + ".dat"
    images, moves = convert_board(file_name, BATCH)
    
    image_name = opath + "input" + str(n)
    move_name = opath + "move" + str(n)
    
    np.savez_compressed(image_name, images)
    np.savez_compressed(move_name, moves)

if __name__ == "__main__":
    args = sys.argv
    ipath = args[1]
    opath = args[2]
    nst = int(args[3])
    ned = int(args[4])
    
    print(args)
    
    for i in range(nst, ned, 1):
        convert_to_np(ipath, opath, i)
