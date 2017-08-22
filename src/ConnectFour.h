#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>

#define CONNECT_FOUR_NUM_ROWS 6
#define CONNECT_FOUR_NUM_COLS 7

enum class ConnectFourStatus {
  X_WINS,
  O_WINS,
  DRAW,
  IN_PROGRESS
};

std::string to_string(ConnectFourStatus status) {
  switch (status) {
    case ConnectFourStatus::X_WINS:
        return "X_WINS";
    case ConnectFourStatus::O_WINS:
        return "O_WINS";
    case ConnectFourStatus::DRAW:
        return "DRAW";
    case ConnectFourStatus::IN_PROGRESS:
        return "IN_PROGRESS";
    }
    return "UNKNOWN";
}

struct ConnectFourAction {
  int column_index;
};

std::string to_string(ConnectFourAction const& action) {
    return "{" + std::to_string(action.column_index) + "}";
}

class ConnectFour {
 public:
  using Action = ConnectFourAction;
  using Status = ConnectFourStatus;
  using BoardStateType = Eigen::Matrix<char,
                                       CONNECT_FOUR_NUM_ROWS,
                                       CONNECT_FOUR_NUM_COLS>;

  ConnectFour(std::string const& state) {
    size_t num_x = std::count(state.begin(), state.end(), 'x');
    size_t num_o = std::count(state.begin(), state.end(), 'o');
    x_turn = !(num_x == num_o + 1);
    memcpy(board_state_.data(), state.c_str(), string_size);
    game_status_ = UpdateConnectFourStatus();
  }

  ConnectFour() {
    Reset();
  }

  BoardStateType GetBoardState() const {
    return board_state_;
  }

  void PrintGame() const {
    std::cout << board_state_ << std::endl << to_string(game_status_) << std::endl;
  }

  void Reset() {
    game_status_ = ConnectFourStatus::IN_PROGRESS;
    board_state_.fill('-');
    x_turn = true;
  }

  std::vector<ConnectFourAction> GetAvailableActions() const {
    std::vector<ConnectFourAction> actions;
    if(GameOver()) {
      return actions;
    }
    for (int col = 0; col < num_cols; ++col) {
      if (board_state_(0, col) == '-') {
        actions.push_back({col});
      }
    }

    return actions;
  }

  void ApplyAction(ConnectFourAction const & action) {
    int row = num_rows - 1;
    while (!(board_state_(row, action.column_index) == '-')) {
      --row;
    }
    board_state_(row, action.column_index) = x_turn ? 'x' : 'o';
    x_turn = !x_turn;
    game_status_ = UpdateConnectFourStatus();
  }

  ConnectFour ForwardModel(ConnectFourAction const& action) const {
    ConnectFour new_board(*this);
    new_board.ApplyAction(action);
    return new_board;
  }

  ConnectFourStatus GetGameStatus() const {
    return game_status_;
  }

  bool GameOver() const {
    return game_status_ != ConnectFourStatus::IN_PROGRESS;
  }

  bool Draw() const {
    return game_status_ == ConnectFourStatus::DRAW;
  }

  std::string GetStateString() const {
    return std::string(board_state_.data(), string_size);
  }

 private:
  bool DidPlayerWin(char player) const {
    // Vertical lines
    for (int row=0; row < num_rows-3; ++row) {
      for (int col=0; col < num_cols; ++col) {
        if (board_state_(row+0, col) == player &&
            board_state_(row+1, col) == player &&
            board_state_(row+2, col) == player &&
            board_state_(row+3, col) == player)
          return true;
      }
    }
    // Horizontal lines
    for (int row=0; row < num_rows; ++row) {
      for (int col=0; col < num_cols-3; ++col) {
        if (board_state_(row, col+0) == player &&
            board_state_(row, col+1) == player &&
            board_state_(row, col+2) == player &&
            board_state_(row, col+3) == player)
          return true;
      }
    }
    // Ascending diagional
    for (int row=0; row < num_rows-3; ++row) {
      for (int col=0; col < num_cols-3; ++col) {
        if (board_state_(row+0, col+0) == player &&
            board_state_(row+1, col+1) == player &&
            board_state_(row+2, col+2) == player &&
            board_state_(row+3, col+3) == player)
          return true;
      }
    }
    // Descending diagional
    for (int row=0; row < num_rows-3; ++row) {
      for (int col=0; col < num_cols-3; ++col) {
        if (board_state_(row+3, col+0) == player &&
            board_state_(row+2, col+1) == player &&
            board_state_(row+1, col+2) == player &&
            board_state_(row+0, col+3) == player)
          return true;
      }
    }
    return false;
  }

  ConnectFourStatus UpdateConnectFourStatus() {
    if (DidPlayerWin('x')) {
      return ConnectFourStatus::X_WINS;
    }
    if (DidPlayerWin('o')) {
      return ConnectFourStatus::O_WINS;
    }

    if (BoardFull()) {
      return ConnectFourStatus::DRAW;
    }

    return ConnectFourStatus::IN_PROGRESS;
  }

  bool BoardFull() const {
    auto actions = GetAvailableActions();
    return (actions.size() == 0);
  }

  BoardStateType board_state_;
  ConnectFourStatus game_status_;
  const int num_rows = CONNECT_FOUR_NUM_ROWS;
  const int num_cols = CONNECT_FOUR_NUM_COLS;
  const int string_size = num_rows * num_cols;
  bool x_turn;
};
