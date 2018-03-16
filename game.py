# coding:utf-8
import numpy as np


class Game:
    def __init__(self, size=4, probability=0.9):
        self.size = size
        self.probability = probability
        self.board = np.zeros((size, size), np.uint32)

    def new_block(self):
        bx, by = np.where(self.board == 0)
        index = np.random.randint(len(bx))
        ix, iy = bx[index], by[index]
        self.board[ix, iy] = 2 if np.random.random() < self.probability else 4

    def move(self, control):
        # 0: left
        # 1: up
        # 2: right
        # 3: down
        assert len(control) == 4
        for c in control:
            if c == 0:
                board = self.board
            elif c == 1:
                pass
            elif c == 2:
                pass
            elif c == 3:
                pass
            else:
                return False
