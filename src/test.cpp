#include <iostream>
#include <memory>
#include <unordered_map>
#include <chrono>

#include <Eigen/Dense>
//a session consists of a game board and two agents
//each agent takes a turn at modifying the board state
//until a terminal state is reached

enum class GameStatus {
  X_WINS,
  O_WINS,
  X_TURN,
  O_TURN,
  DRAW
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
    case GameStatus::DRAW:
      return "DRAW";
  }
  return "UNKNOWN";
}


struct TicTacToeAction {
  int row_index;
  int column_index;
  char value;
};

std::string to_string(TicTacToeAction const& action) {
  return "{" 
  + std::to_string(action.row_index) + ", " 
  + std::to_string(action.column_index) + ", " 
  + action.value + "}";
}

class TicTacToeBoard {
  public:
  using BoardState = Eigen::Matrix<char, 3, 3>;
  TicTacToeBoard() {
    Reset(); 
  };

  BoardState GetBoardState() const {
    return board_state_;
  }
  
  void Reset() {
    game_status_ = GameStatus::O_TURN;
    board_state_.fill('-');
    x_turn_ = false;
  }

  bool IsActionValid(TicTacToeAction const& action) const {
    if(GameOver()) return false;
    bool correct_turn = game_status_ == GameStatus::X_TURN ? action.value == 'x' : action.value == 'o';
    if(!correct_turn) { 
      std::cout << "Wrong Turn" << std::endl;
      return false;
    }
    bool in_bounds = action.column_index < size && action.row_index < size;
    if(!in_bounds) return false;
    bool is_free = board_state_(action.row_index, action.column_index) == '-';
    if(!is_free) {
      std::cout << board_state_(action.row_index, action.column_index) << std::endl;
      std::cout << "Tried to take a move on an occupied space" << std::endl;
      return false;
    }
    return true;
  }

  std::vector<TicTacToeAction> GetAvailableActions() const {
    std::vector<TicTacToeAction> actions;
    char turn_value = game_status_ == GameStatus::X_TURN ? 'x' : 'o';
    for(int row_index = 0; row_index < size; row_index++) {
      for(int column_index = 0; column_index < size; column_index++) {
        if(board_state_(row_index, column_index) == '-') {
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
    } else {
      std::cout << "Tried to apply invalid action " << to_string(action) << std::endl;
      std::cout << board_state_ << std::endl;
    }
  }

  GameStatus GetGameStatus() const {
    return game_status_;
  }

  bool GameOver() const {
    return game_status_ == GameStatus::X_WINS 
        || game_status_ == GameStatus::O_WINS
        || game_status_ == GameStatus::DRAW;
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

    if(BoardFull()) {
      return GameStatus::DRAW;
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
  
  BoardState board_state_;
  GameStatus game_status_;
  const int size = 3;
  bool x_turn_;
};

class TicTacToeAgent {
public:
  TicTacToeAgent(TicTacToeBoard* game) : game_(game) { }
  virtual TicTacToeAction GetAction() const = 0;
protected:
  TicTacToeBoard* game_;
};

class PickFirstActionAgent : TicTacToeAgent {
  public:
  PickFirstActionAgent(TicTacToeBoard* game) : TicTacToeAgent(game) { }
  virtual TicTacToeAction GetAction() const override {
    std::vector<TicTacToeAction> actions = game_->GetAvailableActions();
    if(actions.size() > 0) {
      return actions[0];
    }
    return {0, 0, '?'};
  }
};

class PickRandomActionAgent : TicTacToeAgent {
  public:
  PickRandomActionAgent(TicTacToeBoard* game) : TicTacToeAgent(game) { }
  virtual TicTacToeAction GetAction() const override {
    std::vector<TicTacToeAction> actions = game_->GetAvailableActions();
    if(actions.size() > 0) {
      return actions[rand()%actions.size()];
    }
    return {0, 0, '?'};
  }
};

int main(int argc, char* argv[])  {
  TicTacToeBoard game;
  srand(time(0));
  auto player1 = std::make_shared<PickRandomActionAgent>(&game);
  auto player2 = std::make_shared<PickRandomActionAgent>(&game);
  using namespace std::chrono;

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  int x_wins=0, o_wins=0, draws=0;
  int num_games = 1e6;
  for(int x = 0; x < num_games; x++) {
    game.Reset();
    while(true) {
      game.ApplyAction(player1->GetAction());
      if(game.GameOver()) break;
      game.ApplyAction(player2->GetAction());
      if(game.GameOver()) break;
    }
    switch(game.GetGameStatus()){
      case GameStatus::X_WINS:
        x_wins++;
        break;
      case GameStatus::O_WINS:
        o_wins++;
        break;
      case GameStatus::DRAW:
        draws++;
        break;
    }
    
  }
  
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  std::cout << "Played " << num_games/time_span.count() << " games per second.";
  std::cout << std::endl; 

  std::cout << "x_wins: " << x_wins/(float)num_games*100 << std::endl;
  std::cout << "o_wins: " << o_wins/(float)num_games*100 << std::endl;
  std::cout << "draws: " << draws/(float)num_games*100 << std::endl;
  
}
