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
        self.root.resizable(False, False)
        self.btn_start = tkinter.Button(self.root, font=("Helvetica", 16), text="start", command=self.start)
        self.btn_start.grid(row=0, column=0)
        self.lbl_score = tkinter.Label(self.root, font=("Helvetica", 16), text="Score: 0")
        self.lbl_score.grid(row=0, column=1)
        self.cv = tkinter.Canvas(self.root, width=410, height=410, bd=0, bg="#BAACA0")
        self.cv.grid(row=1, column=0, columnspan=2)
        self.bg_color = {
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
            16384: "#AB60A6",
            32768: "#A755A4"
        }
        self.cv.bind_all("<KeyPress-Up>", self.control)
        self.cv.bind_all("<KeyPress-Down>", self.control)
        self.cv.bind_all("<KeyPress-Left>", self.control)
        self.cv.bind_all("<KeyPress-Right>", self.control)
        self.start()
        self.root.mainloop()

    @staticmethod
    def create_arc_rectangle(cv, x, y, l, r=5, outline=None, fill=None):
        cv.create_rectangle(x, y + r, x + l, y + l - r, outline=outline, fill=fill)
        cv.create_rectangle(x + r, y, x + l - r, y + l, outline=outline, fill=fill)
        cv.create_arc(x, y, x + r * 2, y + r * 2, start=90, extent=90, outline=outline, fill=fill)
        cv.create_arc(x + l - 2 * r, y, x + l, y + r * 2, start=0, extent=90, outline=outline, fill=fill)
        cv.create_arc(x, y + l - 2 * r, x + r * 2, y + l, start=180, extent=90, outline=outline, fill=fill)
        cv.create_arc(x + l - 2 * r, y + l - 2 * r, x + l, y + l, start=270, extent=90, outline=outline, fill=fill)

    @staticmethod
    def create_text(cv, x, y, value):
        if value <= 4:
            color = "#766D65"
        else:
            color = "#F8F5F1"
        cv.create_text(x, y, fill=color, font=("Helvetica", 20), text=str(value))

    def draw_block(self, x, y, v):
        GameGUI.create_arc_rectangle(self.cv, x * 100 + 10, y * 100 + 10, 90, r=5, outline=self.bg_color.get(v, "#EDE3D9"), fill=self.bg_color.get(v, "#EDE3D9"))
        if v > 0:
            GameGUI.create_text(self.cv, x * 100 + 55, y * 100 + 55, v)

    def start(self):
        super(GameGUI, self).__init__()
        self.new_block()
        self.show()

    def control(self, event):
        flag = False
        if event.keysym == "Left":
            flag = self.move([0, -1, -1, -1])
        elif event.keysym == "Up":
            flag = self.move([1, -1, -1, -1])
        elif event.keysym == "Right":
            flag = self.move([2, -1, -1, -1])
        elif event.keysym == "Down":
            flag = self.move([3, -1, -1, -1])
        if flag:
            self.new_block()
        self.show()

    def show(self):
        self.lbl_score.configure(text="Score: %d" % self.score)
        self.cv.delete("all")
        for i in range(4):
            for j in range(4):
                self.draw_block(j, i, self.board[i, j])


if __name__ == '__main__':
    # GameCore().run()
    GameGUI()
