#include <iostream>
#include <Eigen/Dense>
#include <unordered_map>

//a session consists of a game board and two agents
//each agent takes a turn at modifying the board state
//until a terminal state is reached

enum class GameStatus {
  X_WINS,
  O_WINS,
  X_TURN,
  O_TURN
};

std::string to_string(GameStatus status) {
  switch(status) {
    case GameStatus::X_WINS:
      return "X_WINS";
    case GameStatus::O_WINS:
      return "O_WINS";
    case GameStatus::X_TURN:
      return "X_TURN";
    case GameStatus::O_TURN:
      return "O_TURN";
  }
  return "UNKNOWN";
}

struct TicTacToeAction {
  int column_index;
  int row_index;
  char value;
};


template <int size>
class TicTacToeBoard {
  public:
  using BoardState = Eigen::Matrix<char, size, size>;
  TicTacToeBoard() {
    Reset(); 
  };

  BoardState GetBoardState() const {
    return board_state_;
  }
  
  void Reset() {
    game_status_ = GameStatus::X_TURN;
    board_state_.fill('_');
    x_turn_ = true;
  }

  bool IsActionValid(TicTacToeAction const& action) const {
    if(GameOver()) return false;
    bool correct_turn = game_status_ == GameStatus::X_TURN ? action.value == 'x' : action.value == 'o';
    if(!correct_turn) return false;
    bool in_bounds = action.column_index < size && action.row_index < size;
    if(!in_bounds) return false;
    bool is_free = board_state_(action.row_index, action.column_index) == '_';
    if(!is_free) return false;
    return true;
  }

  std::vector<TicTacToeAction> GetAvailableActions() const {
    std::vector<TicTacToeAction> actions;
    char turn_value = game_status_ == GameStatus::X_TURN ? 'x' : 'o';
    for(int row_index = 0; row_index < size; row_index++) {
      for(int column_index = 0; column_index < size; column_index++) {
        if(board_state_(row_index, column_index) == '_') {
          actions.push_back({row_index, column_index, turn_value});
        }
      }
    }
    return actions;
  }

  void ApplyAction(TicTacToeAction const& action) {
    if(IsActionValid(action)) {
      board_state_(action.row_index, action.column_index) = action.value;
      x_turn_ = !x_turn_;
      game_status_ = UpdateGameStatus(); 
    }
  }

  GameStatus GetGameStatus() const {
    return game_status_;
  }

  bool GameOver() const {
    return game_status_ == GameStatus::X_WINS || game_status_ == GameStatus::O_WINS;
  }

  private: 

  GameStatus UpdateGameStatus() {
    for(int row_index = 0; row_index < size; row_index++) {
      if(CheckRowValue(row_index, 'o')) {
        return GameStatus::O_WINS;
      } else if(CheckRowValue(row_index, 'x')) {
        return GameStatus::X_WINS;
      }
    }

    for(int column_index = 0; column_index < size; column_index++) {
      if(CheckColumnValue(column_index, 'o')) {
        return GameStatus::O_WINS;
      } else if(CheckColumnValue(column_index, 'x')) {
        return GameStatus::X_WINS;
      }
    }

    if(CheckFirstDiagonal('x') || CheckSecondDiagonal('x')) {
      return GameStatus::X_WINS;
    } else if(CheckFirstDiagonal('o') || CheckSecondDiagonal('o')) {
      return GameStatus::O_WINS;
    }
  
    return x_turn_ ? GameStatus::X_TURN : GameStatus::O_TURN;
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
  
  BoardState board_state_;
  GameStatus game_status_;
  bool x_turn_;
};

int main(int argc, char* argv[])  {
  TicTacToeBoard<3> game;

  game.ApplyAction({0, 0, 'x'});  
  game.ApplyAction({1, 1, 'x'});  
  game.ApplyAction({2, 2, 'x'});  

  std::cout << game.GetBoardState() << std::endl;
  std::cout << to_string(game.GetGameStatus()) << std::endl;
}
