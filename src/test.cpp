#include <iostream>
#include <Eigen/Dense>
#include <unordered_map>

//a session consists of a game board and two agents
//each agent takes a turn at modifying the board state
//until a terminal state is reached

enum class BoardStatus {
  X_WINS,
  O_WINS,
  IN_PROGRESS
};

std::string to_string(BoardStatus status) {
  switch(status) {
    case BoardStatus::X_WINS:
      return "X_WINS";
    case BoardStatus::O_WINS:
      return "O_WINS";
    case BoardStatus::IN_PROGRESS:
      return "IN_PROGRESS";
  }
  return "UNKNOWN";
}


template <int size>
class TicTacToeBoard {
  public:
  using BoardState = Eigen::Matrix<char, size, size>;
  TicTacToeBoard() { 
    board_state_.fill('_');
    board_state_(2, 0) = 'x';
    board_state_(1, 1) = 'x';
    board_state_(0, 2) = 'x';
  };

  BoardState GetBoardState() {
    return board_state_;
  }

  BoardStatus CheckBoardStatus() {
   //check rows
  for(int row_index = 0; row_index < size; row_index++) {
    if(CheckRowValue(row_index, 'o')) {
      return BoardStatus::O_WINS;
    } else if(CheckRowValue(row_index, 'x')) {
      return BoardStatus::X_WINS;
    }
  }

  //check columns
  for(int column_index = 0; column_index < size; column_index++) {
    if(CheckColumnValue(column_index, 'o')) {
      return BoardStatus::O_WINS;
    } else if(CheckColumnValue(column_index, 'x')) {
      return BoardStatus::X_WINS;
    }
  }

  //check diagonals
  if(CheckFirstDiagonal('x') || CheckSecondDiagonal('x')) {
    return BoardStatus::X_WINS;
  } else if(CheckFirstDiagonal('o') || CheckSecondDiagonal('o')) {
    return BoardStatus::O_WINS;
  }
  
  return BoardStatus::IN_PROGRESS;
  }
  
  bool CheckFirstDiagonal(char value) {
    for(int index = 0; index < size; index++) {
      if(board_state_(index, index) != value) {
        return false;
      }
    }
    return true;
  }

  bool CheckSecondDiagonal(char value) {
    for(int index = 0; index < size; index++) {
      if(board_state_(size - index - 1, index) != value) {
        return false;
      }
    }
    return true;
  }

  bool CheckRowValue(size_t row_index, char value) {
    for(int column_index = 0; column_index < size; column_index++) {
      if(board_state_(row_index, column_index) != value) {
        return false;
      }
    } 
    return true;
  }

  bool CheckColumnValue(size_t column_index, char value) {
    for(int row_index = 0; row_index < size; row_index++) {
      if(board_state_(row_index, column_index) != value) {
        return false;
      }
    } 
    return true;
  }

  private:
  BoardState board_state_;
};



int main(int argc, char* argv[])  {
  TicTacToeBoard<3> game;
  std::cout << game.GetBoardState() << std::endl;
  std::cout << to_string(game.CheckBoardStatus()) << std::endl;
}
