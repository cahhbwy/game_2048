# coding:utf-8
import numpy as np
import tkinter


class GameCore:
    def __init__(self, size=4, probability=0.9):
        self.size = size
        self.probability = probability
        self.board = np.zeros((size, size), np.uint32)
        self.score = 0

    def new_block(self):
        bx, by = np.where(self.board == 0)
        index = np.random.randint(len(bx))
        ix, iy = bx[index], by[index]
        self.board[ix, iy] = 2 if np.random.random() < self.probability else 4

    def _move(self, board):
        moved = False
        for line in board:
            p = 0
            q = 1
            while q < self.size:
                if line[p] == 0:
                    if line[q] == 0:
                        q += 1
                    else:
                        moved = True
                        line[p] = line[q]
                        line[q] = 0
                        q += 1
                else:
                    if line[p] == line[q]:
                        moved = True
                        line[p] *= 2
                        self.score += line[p]
                        line[q] = 0
                        p += 1
                        q += 1
                    elif line[q] == 0:
                        q += 1
                    else:
                        p += 1
                        q = p + 1
        return moved, board

    def move(self, control):
        # 0: left
        # 1: up
        # 2: right
        # 3: down
        assert len(control) == 4
        for c in control:
            if c == 0:
                board = np.rot90(self.board, 0)
                moved, board = self._move(board)
                if moved:
                    self.board = np.rot90(board, 4)
                    return True
            elif c == 1:
                board = np.rot90(self.board, 1)
                moved, board = self._move(board)
                if moved:
                    self.board = np.rot90(board, 3)
                    return True
            elif c == 2:
                board = np.rot90(self.board, 2)
                moved, board = self._move(board)
                if moved:
                    self.board = np.rot90(board, 2)
                    return True
            elif c == 3:
                board = np.rot90(self.board, 3)
                moved, board = self._move(board)
                if moved:
                    self.board = np.rot90(board, 1)
                    return True
            else:
                return False

    def show(self):
        for i in self.board:
            for j in i:
                if j != 0:
                    print("%6d" % j, end="")
                else:
                    print("  ____", end="")
            print()
        print("Score: %d" % self.score)

    def alive(self):
        score = self.score
        board = self.board.copy()
        if self.move([0, 1, 2, 3]):
            self.score = score
            self.board = board
            return True
        else:
            return False

    def run(self):
        ct = {"4": 0, "8": 1, "6": 2, "5": 3}
        self.new_block()
        while self.alive():
            self.show()
            op = input()
            if self.move([ct.get(op, -1), -1, -1, -1]):
                self.new_block()
        self.show()


class GameGUI(GameCore):
    def __init__(self):
        super(GameGUI, self).__init__()
        self.root = tkinter.Tk()
        self.cv = tkinter.Canvas(self.root, width=408, height=408, bg="#BAACA0")
        self.color_set = {
            0: "#CCBFB4",
            2: "#EDE3D9",
            4: "#ECDFC8",
            8: "#F1B079",
            16: "#F49463",
            32: "#F57B5F",
            64: "#F55D3B",
            128: "#ECCE73",
            256: "#ECCB63",
            512: "#ECC752",
            1024: "#ECC443",
            2048: "#ECC133",
            4096: "#B884AB",
            8192: "#AF6CA8",
        }
        self.cv.pack()
        self.draw_block(0, 0, 2)
        self.draw_block(0, 1, 4)
        self.draw_block(0, 2, 8)
        self.draw_block(0, 3, 16)
        self.draw_block(1, 0, 32)
        self.draw_block(1, 1, 64)
        self.draw_block(1, 2, 128)
        self.draw_block(1, 3, 256)
        self.draw_block(2, 0, 512)
        self.draw_block(2, 1, 1024)
        self.draw_block(2, 2, 2048)
        self.draw_block(2, 3, 4096)
        self.draw_block(3, 0, 8192)
        self.draw_block(3, 1, 16384)
        self.draw_block(3, 2, 0)
        self.draw_block(3, 3, 0)

        self.root.mainloop()

    def draw_block(self, x, y, v):
        self.cv.create_rectangle(x * 100 + 10, y * 100 + 10, x * 100 + 100, y * 100 + 100, fill=self.color_set.get(v, "#EDE3D9"), width=0)


if __name__ == '__main__':
    # GameCore().run()
    GameGUI()
