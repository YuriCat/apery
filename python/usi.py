# -*- coding: utf-8 -*-
# usi.py
# Katsuki Ohto

import copy, math, time
from enum import IntEnum

import numpy as np

from shogi import *

def do_usi_command_loop():

    while True:

        message = input().split(' ')

        if len(message) <= 0:
            continue

        command = message[0]

        if commad == 'go':
