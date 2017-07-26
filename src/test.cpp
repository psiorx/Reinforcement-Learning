#include <iostream>
#include <memory>
#include <unordered_map>
#include <set>
#include <chrono>

#include <Eigen/Dense>
//a session consists of a game board and two agents
//each agent takes a turn at modifying the board state
//until a terminal state is reached

enum class GameStatus {
  WIN,
  DRAW,
  IN_PROGRESS
};

std::string to_string(GameStatus status) {
  switch(status) {
    case GameStatus::WIN:
      return "WINS";
    case GameStatus::DRAW:
      return "DRAW";
    case GameStatus::IN_PROGRESS:
      return "IN_PROGRESS";
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
    game_status_ = GameStatus::IN_PROGRESS;
    board_state_.fill('-');
  }

  std::vector<TicTacToeAction> GetAvailableActions(char value) const {
    std::vector<TicTacToeAction> actions;
    for(int row_index = 0; row_index < size; row_index++) {
      for(int column_index = 0; column_index < size; column_index++) {
        if(board_state_(row_index, column_index) == '-') {
          actions.push_back({row_index, column_index, value});
        }
      }
    }
    return actions;
  }

  void ApplyAction(TicTacToeAction const& action) {
    board_state_(action.row_index, action.column_index) = action.value;
    game_status_ = UpdateGameStatus(); 
  }

  GameStatus GetGameStatus() const {
    return game_status_;
  }

  bool GameOver() const {
    return game_status_ != GameStatus::IN_PROGRESS;
  }
  
  std::string GetStateString() const {
    return std::string(board_state_.data(), string_size);
  }

  private: 

  GameStatus UpdateGameStatus() {
    for(int row_index = 0; row_index < size; row_index++) {
      if(CheckRowValue(row_index, 'o')) {
        return GameStatus::WIN;
      } else if(CheckRowValue(row_index, 'x')) {
        return GameStatus::WIN;
      }
    }

    for(int column_index = 0; column_index < size; column_index++) {
      if(CheckColumnValue(column_index, 'o')) {
        return GameStatus::WIN;
      } else if(CheckColumnValue(column_index, 'x')) {
        return GameStatus::WIN;
      }
    }

    if(CheckFirstDiagonal('x') || CheckSecondDiagonal('x')) {
      return GameStatus::WIN;
    } else if(CheckFirstDiagonal('o') || CheckSecondDiagonal('o')) {
      return GameStatus::WIN;
    }

    if(BoardFull()) {
      return GameStatus::DRAW;
    }
  
    return GameStatus::IN_PROGRESS;
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
  const int string_size = size * size;
};

class TicTacToeAgent {
public:
  TicTacToeAgent(TicTacToeBoard* game, char which_player) 
	  : game_(game), which_player_(which_player) { }
  virtual TicTacToeAction GetAction() const = 0;
protected:
  TicTacToeBoard* game_;
  char which_player_;
};

class PickFirstActionAgent : TicTacToeAgent {
  public:
  PickFirstActionAgent(TicTacToeBoard* game, char which_player) : TicTacToeAgent(game, which_player) { }
  virtual TicTacToeAction GetAction() const override {
    std::vector<TicTacToeAction> actions = game_->GetAvailableActions(which_player_);
    if(actions.size() > 0) {
      return actions[0];
    }
    return {0, 0, '?'};
  }
};

class PickRandomActionAgent : TicTacToeAgent {
  public:
  PickRandomActionAgent(TicTacToeBoard* game, char which_player) : TicTacToeAgent(game, which_player) { }
  virtual TicTacToeAction GetAction() const override {
    std::vector<TicTacToeAction> actions = game_->GetAvailableActions(which_player_);
    if(actions.size() > 0) {
      return actions[rand()%actions.size()];
    }
    return {0, 0, '?'};
  }
};

int main(int argc, char* argv[])  {
  TicTacToeBoard game;
  srand(time(0));
  auto x_player = std::make_shared<PickRandomActionAgent>(&game, 'x');
  auto o_player = std::make_shared<PickRandomActionAgent>(&game, 'o');
  using namespace std::chrono;

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  int x_wins=0, o_wins=0, draws=0;
  int num_games = 1e6;

  std::set<std::string> game_states;
  
  for(int x = 0; x < num_games; x++) {
    game.Reset();
    while(true) {
      game.ApplyAction(x_player->GetAction());
      if(game.GameOver()) break;
      game.ApplyAction(o_player->GetAction());
      if(game.GameOver()) break;
    }
    game_states.insert(game.GetStateString());
  }
 
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

  //for(auto const& state : game_states) std::cout << state << std::endl;
  std::cout << "Played " << num_games/time_span.count() << " games per second." << std::endl;
  std::cout << "x_wins: " << x_wins/(float)num_games*100 << std::endl;
  std::cout << "o_wins: " << o_wins/(float)num_games*100 << std::endl;
  std::cout << "draws: " << draws/(float)num_games*100 << std::endl;
   
  std::cout << game_states.size() << std::endl;
  
}
