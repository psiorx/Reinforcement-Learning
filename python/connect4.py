import numpy as np

class Connect4:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.player = 1
        self.game_over = False            

    def take_action(self, column):      
        open_row = self.get_open_row(column)
        if open_row >= 0:
            self.board[open_row, column] = self.player
            self.player = 1 if self.player == 2 else 2
            if self.is_draw():
                self.game_over = True
                return 0
            if self.is_win((open_row, column)):
                return 1 if self.player == 2 else -1
                self.game_over = True
        return 0

    def count_matches(self, index, delta):
        count = 0
        current_index = index
        center_value = self.board[current_index]
        for i in range(3):
            current_index = (current_index[0] + delta[0], current_index[1] + delta[1])
            if not self.in_map(current_index):
                return count
            if self.board[current_index] == center_value:
                count = count + 1
            else:
                return count
        return count

    def in_map(self, index):
        if index[0] < 0 or index[0] >= 6:
            return False
        if index[1] < 0 or index[1] >= 7:
            return False
        return True

    def is_draw(self):
        return len(self.get_valid_actions()) == 0

    def is_win(self, position):

        horizontal_matches = 1 + self.count_matches(position, (1, 0)) + self.count_matches(position, (-1, 0))
        if horizontal_matches >= 4:
            return True
        
        vertical_matches = 1 + self.count_matches(position, (0, 1)) + self.count_matches(position, (0, -1))
        if vertical_matches >= 4:
            return True

        diagonal_matches = 1 + self.count_matches(position, (1, 1)) + self.count_matches(position, (-1, -1))
        if vertical_matches >= 4:
            return True

        return False

    def get_valid_actions(self):
        matches= np.where(self.board[0, :] == 0)
        return matches[0]

    def get_open_row(self, column):
        open_row = np.nonzero(self.board[:, column])
        if len(open_row[0]) > 0:
            return open_row[0][0] - 1
        else:
            return 5