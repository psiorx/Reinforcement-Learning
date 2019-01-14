#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>

enum class TicTacToeStatus {
    X_WINS,
    O_WINS,
    DRAW,
    IN_PROGRESS
};

std::string to_string(TicTacToeStatus status) {
    switch(status) {
        case TicTacToeStatus::X_WINS:
            return "X_WINS";
        case TicTacToeStatus::O_WINS:
            return "O_WINS";
        case TicTacToeStatus::DRAW:
            return "DRAW";
        case TicTacToeStatus::IN_PROGRESS:
            return "IN_PROGRESS";
    }
    return "UNKNOWN";
}

struct TicTacToeAction {
    int row_index;
    int column_index;
    
    bool operator==(const TicTacToeAction &other) const
    { return (row_index == other.row_index
            && column_index == other.column_index);
    }
};

namespace std
{
    template <>
    struct hash<TicTacToeAction> {
        size_t operator()( const TicTacToeAction& k ) const {
            size_t res = 17;
            res = res * 31 + hash<int>()( k.row_index );
            res = res * 31 + hash<int>()( k.column_index );
            return res;
        }
    };
}

std::string to_string(TicTacToeAction const& action) {
    return "{"
           + std::to_string(action.row_index) + ", "
           + std::to_string(action.column_index) + "}";
}

class TicTacToe {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Action = TicTacToeAction;
    using Status = TicTacToeStatus;
    using BoardStateType = Eigen::Matrix<char, 3, 3>;

    TicTacToe(std::string const& state) {
        size_t num_x = std::count(state.begin(), state.end(), 'x');
        size_t num_o = std::count(state.begin(), state.end(), 'o');
        x_turn = !(num_x == num_o + 1);
        memcpy(board_state_.data(), state.c_str(), 9);
        game_status_ = UpdateTicTacToeStatus();
    }

    TicTacToe() {
        Reset();
    };

    BoardStateType GetBoardState() const {
        return board_state_;
    }

    float GetReward() const {
        if(GameOver() && !Draw()) {
            return x_turn ? -1.0f : 1.0f;
        }
        return 0.0f;
    }

    bool FirstPlayersTurn() const {
        return x_turn;
    }

    void Reset() {
        game_status_ = TicTacToeStatus::IN_PROGRESS;
        board_state_.fill('-');
        x_turn = true;
    }

    std::vector<TicTacToeAction> GetAvailableActions() const {
        std::vector<TicTacToeAction> actions;
        if(GameOver()) {
          return actions;
        }
        for(int row_index = 0; row_index < size; row_index++) {
            for(int column_index = 0; column_index < size; column_index++) {
                if(board_state_(row_index, column_index) == '-') {
                    actions.push_back({row_index, column_index});
                }
            }
        }
        return actions;
    }

    float ApplyAction(TicTacToeAction const& action) {
        board_state_(action.row_index, action.column_index) = x_turn ? 'x' : 'o';
        x_turn = !x_turn;
        game_status_ = UpdateTicTacToeStatus();
        return GetReward();
    }

    TicTacToe ForwardModel(TicTacToeAction const& action) const {
        TicTacToe new_board(*this);
        new_board.ApplyAction(action);
        return new_board;
    }

    TicTacToeStatus GetGameStatus() const {
        return game_status_;
    }

    bool GameOver() const {
        return game_status_ != TicTacToeStatus::IN_PROGRESS;
    }

    bool Draw() const {
        return game_status_ == TicTacToeStatus::DRAW;
    }

    std::string GetStateString() const {
        return std::string(board_state_.data(), string_size);
    }

private:

    TicTacToeStatus UpdateTicTacToeStatus() {
        for(int row_index = 0; row_index < size; row_index++) {
            if(CheckRowValue(row_index, 'o')) {
                return TicTacToeStatus::O_WINS;
            } else if(CheckRowValue(row_index, 'x')) {
                return TicTacToeStatus::X_WINS;
            }
        }

        for(int column_index = 0; column_index < size; column_index++) {
            if(CheckColumnValue(column_index, 'o')) {
                return TicTacToeStatus::O_WINS;
            } else if(CheckColumnValue(column_index, 'x')) {
                return TicTacToeStatus::X_WINS;
            }
        }

        if(CheckFirstDiagonal('x') || CheckSecondDiagonal('x')) {
            return TicTacToeStatus::X_WINS;
        } else if(CheckFirstDiagonal('o') || CheckSecondDiagonal('o')) {
            return TicTacToeStatus::O_WINS;
        }

        if(BoardFull()) {
            return TicTacToeStatus::DRAW;
        }

        return TicTacToeStatus::IN_PROGRESS;
    }

    bool CheckFirstDiagonal(char value) const {
        for(int index = 0; index < size; index++) {
            if(board_state_(index, index) != value) {
                return false;
            }
        }
        return true;
    }

    bool CheckSecondDiagonal(char value) const {
        for(int index = 0; index < size; index++) {
            if(board_state_(size - index - 1, index) != value) {
                return false;
            }
        }
        return true;
    }

    bool CheckRowValue(size_t row_index, char value) const {
        for(int column_index = 0; column_index < size; column_index++) {
            if(board_state_(row_index, column_index) != value) {
                return false;
            }
        }
        return true;
    }

    bool CheckColumnValue(size_t column_index, char value) const {
        for(int row_index = 0; row_index < size; row_index++) {
            if(board_state_(row_index, column_index) != value) {
                return false;
            }
        }
        return true;
    }

    bool BoardFull() const {
        for(int row_index = 0; row_index < size; row_index++) {
            for(int column_index = 0; column_index < size; column_index++) {
                if(board_state_(row_index, column_index) == '-') {
                    return false;
                }
            }
        }
        return true;
    }

    BoardStateType board_state_;
    TicTacToeStatus game_status_;
    const int size = 3;
    const int string_size = size * size;
    bool x_turn;
};