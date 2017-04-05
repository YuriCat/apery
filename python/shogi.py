# -*- coding: utf-8 -*-
# shogi.py
# Katsuki Ohto

import copy, math, time

import numpy as np


ROWS = 9
COLS = 9
SIZE = ROWS * COLS

# color
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




