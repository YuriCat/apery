# -*- coding: utf-8 -*-
# shogi.py
# Katsuki Ohto

import copy, math, time
from enum import IntEnum

import numpy as np

# board
FILE_NUM = 9
RANK_NUM = 9
SQUARE_NUM = FILE_NUM * RANK_NUM

def make_file(sq):
    return sq // RANK_NUM
def make_rank(sq):
    return sq % RANK_NUM
def make_square(f, r):
    return f * RANK_NUM + r

DELTA_N = -1
DELTA_E = -RANK_NUM
DELTA_S = 1
DELTA_W = RANK_NUM

DELTA_NE = DELTA_N + DELTA_E
DELTA_SE = DELTA_S + DELTA_E
DELTA_SW = DELTA_S + DELTA_W
DELTA_NW = DELTA_N + DELTA_W

# color
class Color(IntEnum):
    BLACK = 0
    WHITE = 1

colorChar = " BW/"
xChar = "ABCDEFGHJKLMNOPQRSTUVWXYZ"

def to_turn_color(t):
    return (t % 2) + BLACK;

def flip_color(c):
    return BLACK + WHITE - c

def char_to_ix(c):
    return xChar.find(c)

# rule
VALID = 0
LEGAL = 0
DOUBLE = 1
SUICIDE = 2
KO = 3
OUT = 4
EYE = 8

# piece
class PieceType(IntEnum):
    NONE = 0
    PAWN = 1
    LANCE = 2
    KNIGHT = 3
    SILVER = 4
    BISHOP = 5
    ROOK = 6
    GOLD = 7
    KING = 8
    PRO_PAWN = 9
    PRO_LANCE = 10
    PRO_KNIGHT = 11
    PRO_SILVER = 12
    HORSE = 13
    DRADON = 14

# piece with color
class Piece(IntEnum):
    EMPTY = 0

    B_PAWN = 1
    B_LANCE = 2
    B_KNIGHT = 3
    B_SILVER = 4
    B_BISHOP = 5
    B_ROOK = 6
    B_GOLD = 7
    B_KING = 8
    B_PRO_PAWN = 9
    B_PRO_LANCE = 10
    B_PRO_KNIGHT = 11
    B_PRO_SILVER = 12
    B_HORSE = 13
    B_DRADON = 14

    W_PAWN = 17
    W_LANCE = 18
    W_KNIGHT = 19
    W_SILVER = 20
    W_BISHOP = 21
    W_ROOK = 22
    W_GOLD = 23
    W_KING = 24
    W_PRO_PAWN = 25
    W_PRO_LANCE = 26
    W_PRO_KNIGHT = 27
    W_PRO_SILVER = 28
    W_HORSE = 29
    W_DRADON = 30

    WALL = 31

    PROMOTE = 8

# usi piece
USI_PIECE_TYPE_STR = ["", "P", "L", "N", "S", "B", "R", "G", "K",
                      "+P", "+L", "+N", "+S", "+B", "+R"]
USI_PIECE_STR = ["",
                 "P", "L", "N", "S", "B", "R", "G", "K",
                 "+P", "+L", "+N", "+S", "+B", "+R", "", "",
                 "p", "l", "n", "s", "b", "r", "g", "k",
                 "+p", "+l", "+n", "+s", "+b", "+r", "#"]

# usi hand piece
USI_HAND_PIECE_STR = ["", "P*", "L*", "N*", "S*", "B*", "R*", "G*"]

def char_to_piece_usi(c):
    return Piece(USI_PIECE_STR.index(c))

# csa piece
CSA_PIECE_TYPE_STR = ["", "FU", "KY", "KE", "GI", "KA", "HI", "KI", "OU",
                      "TO", "NY", "NK", "NG", "UM", "RY"]
CSA_PIECE_STR = [" * ",
                 "+FU", "+KY", "+KE", "+GI", "+KA", "+HI", "+KI", "+OU",
                 "+TO", "+NY", "+NK", "+NG", "+UM", "+RY", "", "",
                 "-FU", "-KY", "-KE", "-GI", "-KA", "-HI", "-KI", "-OU",
                 "-TO", "-NY", "-NK", "-NG", "-UM", "-RY"]

SFEN_HIRATE = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

class MinimumMove:
    def __init__(self, sq_from, sq_to, promotion):
        self.sq_from = sq_from
        self.sq_to = sq_to
        self.promotion = promotion

    def __str__(self):
        if self.is_drop():
            return "(" + USI_HAND_PIECE_STR[self.piece_type_dropped()] + "," + str(self.sq_to) + ")"
        else:
            if self.is_promotion():
                return "(" + str(self.sq_from) + "," + str(self.sq_to) + "+)"
            else:
                return "(" + str(self.sq_from) + "," + str(self.sq_to) + ")"

    def is_promotion(self):
        return self.promotion

    def is_drop(self):
        return self.sq_from >= 121

    def piece_type_dropped(self):
        if self.is_drop():
            return PieceType(self.sq_from - 121 + PieceType.PAWN)
        else:
            return None

class Move:
    def __init__(self, sq_from, sq_to, promotion):
        self.sq_from = sq_from
        self.sq_to = sq_to
        self.promotion = promotion

    def __str__(self):
        # usi
        if self.is_drop():
           return HAND_PIECE_STR[self.piece]

    def is_promotion(self):
        return self.promotion

    def is_drop(self):
        return self.sq_from >= SQUARE_NUM

# piece
class HandPieceType(IntEnum):
    PAWN = 0
    LANCE = 1
    KNIGHT = 2
    SILVER = 3
    BISHOP = 4
    ROOK = 5
    GOLD = 6
    NUM = 7

def piece_to_hand_piece(piece):
    return HandPiece(piece - Piece.PAWN + HandPiece.PAWN)
def hand_piece_to_piece(hp):
    return Piece(hp - HandPieceType.PAWN + Piece.PAWN)

class Hand:
    def __init__(self):
        self.quantity = np.zeros(HandPieceType.NUM, dtype = HandPieceType)
    def clear(self):
        self.quantity.fill(0)
    def set_piece(self, pt):
        self.quantity[pt] += 1

# hash value table for board
BOARD_HASH_TABLE = np.empty((121, 32), dtype = np.uint64)
HAND_HASH_TABLE = np.empty((2, 7, 41), dtype = np.uint64)

np.random.seed(12345679)
for i in range(121):
    BOARD_HASH_TABLE[i][0] = 0 # empty
    for p in range(1, 32):
        BOARD_HASH_TABLE[i][p] = np.random.random_integers(2 ** 62).astype(np.uint64)

class Position:
    def __init__(self):
        self.padding = 1
        self.file_num = FILE_NUM + padding * 2
        self.rank_num = RANK_NUM + padding * 2
        self.square_num = self.file_num * self.rank_num
        self.board = np.zeros((self.square_num), dype = Piece)
        self.clear_board()
        self.hand = [Hand(), Hand()]
        self.set_sfen(SFEN_HIRATE)
        self.game_ply = 0
        self.board_key = self.compute_board_key()
        self.hand_key = self.compute_hand_key()
        self.material = self.compute_material()

    def psq(self, sq):
        return (make_file(sq) + padding) * self.rank_num + (make_rank(sq) + padding)

    def clear_board(self):
        self.board.fill(Piece.WALL)
        for sq in range(SQUARE_NUM):
            self.board[self.psq(sq)] = Piece.EMPTY
    def clear_hand(self):
        for h in hand:
            h.clear()

    def set_piece(self, piece, sq):
        self.board[self.psq(sq)] = piece

    def set_hand(self, piece):
        hand[piece_to_color(piece)].set_piece(piece_to_piece_type(piece))

    def incorrect_sfen(self, sfen_str):
        print("incorrect SFEN string : %s", sfen_str)
        return None

    def set_sfen(self, sfen_str):
        sfen_str_list = sfen_str.split(' ')
        sq = make_square(FILE_NUM - 1, 0)
        promote_flag = 0
        for c in sfen_str_list[0]:
            if c.isdigit():
                sq += DELTA_E * int(c)
            elif c == '/':
                sq += DELTA_W * RANK_NUM + DELTA_S
            elif c == '+':
                promote_flag = Piece.PROMOTE
            elif c in USI_PIECE_STR:
                if 0 <= sq and sq < SQUARE_NUM:
                    self.set_piece(USI_PIECE_STR.index(c) + promote_flag)
                    promote_flag = 0
                    sq += DELTA_E
                else:
                    return self.incorrect_sfen(sfen)
            else:
                return self.incorrect_sfen(sfen)

        digits = 0
        for c in sfen_str_list[1]:
            if c == '-':
                self.clear_hand()
            elif c.isdigit():
                digits = digits * 10 + int(c)
            elif c in USI_PIECE_STR:
                self.set_hand(USI_PIECE_STR.index(c), max(1, digits))
                digits = 0
            else:
                return self.incorrect_sfen(sfen)

        self.game_ply = int(sfen_str_list[2])





if __name__ == '__main__':

    m = MinimumMove(15, 20, True)
    print(m)

    m = MinimumMove(121 + 3, 50, False)
    print(m)