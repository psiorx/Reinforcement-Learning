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
};

std::string to_string(TicTacToeAction const& action) {
    return "{"
           + std::to_string(action.row_index) + ", "
           + std::to_string(action.column_index) + "}";
}

class TicTacToe {
public:
    using Action = TicTacToeAction;
    using Status = GameStatus;
    using BoardStateType = Eigen::Matrix<char, 3, 3>;
    TicTacToe() {
        Reset();
    };

    BoardStateType GetBoardState() const {
        return board_state_;
    }

    void Reset() {
        game_status_ = GameStatus::IN_PROGRESS;
        board_state_.fill('-');
        x_turn = true;
    }

    std::vector<TicTacToeAction> GetAvailableActions() const {
        std::vector<TicTacToeAction> actions;
        for(int row_index = 0; row_index < size; row_index++) {
            for(int column_index = 0; column_index < size; column_index++) {
                if(board_state_(row_index, column_index) == '-') {
                    actions.push_back({row_index, column_index});
                }
            }
        }
        return actions;
    }

    void ApplyAction(TicTacToeAction const& action) {
        board_state_(action.row_index, action.column_index) = x_turn ? 'x' : 'o';
        x_turn = !x_turn;
        game_status_ = UpdateGameStatus();
    }

    TicTacToe ForwardModel(TicTacToeAction const& action) const {
        TicTacToe new_board(*this);
        new_board.ApplyAction(action);
        return new_board;
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
    const int string_size = size * size + 1;
    bool x_turn;
};

template <class Game, template <class Game> class Agent1, template <class Game> class Agent2>
class GameSession {
public:
    typename Game::Status PlayOnce() {
        while(true) {
            game.ApplyAction(player1.GetAction(game));
            if(game.GameOver()) break;
            game.ApplyAction(player2.GetAction(game));
            if(game.GameOver()) break;
        }
        return game.GetGameStatus();
    }

    std::vector<typename Game::Status> PlayN(size_t n) {
        std::vector<typename Game::Status> status_results;
        status_results.reserve(n);
        for(size_t count = 0; count < n ; count++) {
            status_results.push_back(PlayOnce());
            Reset();
        }
        return status_results;
    }

    void Reset() {
        game.Reset();
    }

private:
    Game game;

    Agent1<Game> player1;
    Agent2<Game> player2;
};

template <class Game>
class PickFirstActionAgent {
public:
    typename Game::Action GetAction(const Game& state) const {
        for(auto const& action : state.GetAvailableActions()) {
            return action;
        }
    }
};

template <class Game>
class PickRandomActionAgent {

    template<typename Iter, typename RandomGenerator>
    Iter select_randomly(Iter start, Iter end, RandomGenerator& g) const {
        std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
        std::advance(start, dis(g));
        return start;
    }

    template<typename Iter>
    Iter select_randomly(Iter start, Iter end) const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return select_randomly(start, end, gen);
    }

public:
    typename Game::Action GetAction(const Game& state) const {
        auto actions = state.GetAvailableActions();
        return *select_randomly(actions.begin(), actions.end());
    }
};

int main(int argc, char* argv[])  {
    using namespace std::chrono;
    int x_wins=0, o_wins=0, draws=0;
    int num_games = 1e6;

    GameSession<TicTacToe, PickRandomActionAgent, PickRandomActionAgent> session;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    for(GameStatus status : session.PlayN(num_games)) {
        switch(status) {
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
