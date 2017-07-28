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
  X_WINS,
  O_WINS,
  DRAW,
  IN_PROGRESS
};

std::string to_string(GameStatus status) {
  switch(status) {
    case GameStatus::X_WINS:
      return "X_WINS";
    case GameStatus::O_WINS:
      return "O_WINS";
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
  using BoardStateType = Eigen::Matrix<char, 3, 3>;
  TicTacToeBoard() {
    Reset(); 
  };

  BoardStateType GetBoardState() const {
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
  
  BoardStateType board_state_;
  GameStatus game_status_;
  const int size = 3;
  const int string_size = size * size;
};

class TicTacToeAgent {
public:
  virtual TicTacToeAction GetAction(TicTacToeBoard* board, char mark) const = 0;
  virtual void TakeAction(TicTacToeBoard* board, char mark) {
    board->ApplyAction(GetAction(board, mark));
  }
};

class PickFirstActionAgent : public TicTacToeAgent {
public:
  virtual TicTacToeAction GetAction(TicTacToeBoard* board, char mark) const override {
    std::vector<TicTacToeAction> actions = board->GetAvailableActions(mark);
    if(actions.size() > 0) {
      return actions[0];
    }
    return {0, 0, '?'};
  }
};

class PickRandomActionAgent : public TicTacToeAgent {
  public:
  virtual TicTacToeAction GetAction(TicTacToeBoard* board, char mark) const override {
    std::vector<TicTacToeAction> actions = board->GetAvailableActions(mark);
    if(actions.size() > 0) {
      return actions[rand()%actions.size()];
    }
    return {0, 0, '?'};
  }
};

template <class Agent1, class Agent2>
class TicTacToeGame {
public:
  TicTacToeGame() : x_player(new Agent1()), o_player(new Agent2()) { 
    Reset();
  }

  GameStatus Play() {
    Reset();
    while(true) {
     TakeTurn();
     if(game_board.GameOver()) {
       return game_board.GetGameStatus();
     }
    }
  }

  void TakeTurn() {
    !(turn_count & 1) ? x_player->TakeAction(&game_board, 'x') : o_player->TakeAction(&game_board, 'o');
    x_turn = !x_turn;
    turn_count++;
  }

  void Reset() {
    game_board.Reset();
    turn_count = 0;
  }

private:
  std::unique_ptr<TicTacToeAgent> x_player;
  std::unique_ptr<TicTacToeAgent> o_player;
  bool x_turn;
  int turn_count;
  TicTacToeBoard game_board;
};

int main(int argc, char* argv[])  {
  srand(time(0));
  using namespace std::chrono;

  TicTacToeGame<PickRandomActionAgent, PickRandomActionAgent> game;

  int x_wins=0, o_wins=0, draws=0;
  int num_games = 1e6;
  
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  for(int x = 0; x < num_games; x++) {
    GameStatus status = game.Play();
    switch(status){
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

  std::cout << "Played " << num_games/time_span.count() << " games per second." << std::endl;
  std::cout << "x_wins: " << x_wins/(float)num_games*100 << std::endl;
  std::cout << "o_wins: " << o_wins/(float)num_games*100 << std::endl;
  std::cout << "draws: " << draws/(float)num_games*100 << std::endl;
}
