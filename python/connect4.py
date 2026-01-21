import numpy as np

class Connect4:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.player = 1
        self.game_over = False
        self.draw = False
        self.winner = 0

    def take_action(self, column):      
        open_row = self.get_open_row(column)
        if open_row >= 0:
            self.board[open_row, column] = self.player
            if self.is_win((open_row, column)):
                self.game_over = True
                self.draw = False
                self.winner = self.player
            elif self.is_draw():
                self.game_over = True
                self.draw = True
            self.player = 1 if self.player == 2 else 2
        else:
            print("invalid move: ")
            print(self.get_valid_actions())
            print(column)

    def get_reward(self, player):
        if self.draw or not self.game_over:
            return 1e-4 if self.draw else 0

        return 1 if self.winner == player else -1
        
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
        if diagonal_matches >= 4:
            return True

        diagonal_matches = 1 + self.count_matches(position, (1, -1)) + self.count_matches(position, (-1, 1))
        if diagonal_matches >= 4:
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

    @staticmethod
    def get_valid_actions_from(board):
        matches = np.where(board[0, :] == 0)
        return matches[0]

    @staticmethod
    def get_open_row_from(board, column):
        open_row = np.nonzero(board[:, column])
        if len(open_row[0]) > 0:
            return open_row[0][0] - 1
        return 5

    @staticmethod
    def in_map_index(index):
        if index[0] < 0 or index[0] >= 6:
            return False
        if index[1] < 0 or index[1] >= 7:
            return False
        return True

    @classmethod
    def count_matches_on_board(cls, board, index, delta):
        count = 0
        current_index = index
        center_value = board[current_index]
        for _ in range(3):
            current_index = (current_index[0] + delta[0], current_index[1] + delta[1])
            if not cls.in_map_index(current_index):
                return count
            if board[current_index] == center_value:
                count = count + 1
            else:
                return count
        return count

    @classmethod
    def is_win_on_board(cls, board, position):
        horizontal_matches = 1 + cls.count_matches_on_board(board, position, (1, 0)) + cls.count_matches_on_board(board, position, (-1, 0))
        if horizontal_matches >= 4:
            return True

        vertical_matches = 1 + cls.count_matches_on_board(board, position, (0, 1)) + cls.count_matches_on_board(board, position, (0, -1))
        if vertical_matches >= 4:
            return True

        diagonal_matches = 1 + cls.count_matches_on_board(board, position, (1, 1)) + cls.count_matches_on_board(board, position, (-1, -1))
        if diagonal_matches >= 4:
            return True

        diagonal_matches = 1 + cls.count_matches_on_board(board, position, (1, -1)) + cls.count_matches_on_board(board, position, (-1, 1))
        if diagonal_matches >= 4:
            return True

        return False

    @classmethod
    def apply_action_to_board(cls, board, player, column):
        new_board = board.copy()
        open_row = cls.get_open_row_from(new_board, column)
        if open_row < 0:
            return new_board, player, True, False, 0
        new_board[open_row, column] = player
        winner = 0
        draw = False
        game_over = False
        if cls.is_win_on_board(new_board, (open_row, column)):
            game_over = True
            winner = player
        elif len(cls.get_valid_actions_from(new_board)) == 0:
            game_over = True
            draw = True
        next_player = 1 if player == 2 else 2
        return new_board, next_player, game_over, draw, winner
