#include <iostream>
#include <memory>
#include <unordered_map>
#include <set>
#include <chrono>

#include <Eigen/Dense>

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

template<typename Iter>
Iter select_randomly(Iter start, Iter end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
}

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
};

std::string to_string(TicTacToeAction const& action) {
    return "{"
           + std::to_string(action.row_index) + ", "
           + std::to_string(action.column_index) + "}";
}

class TicTacToe {
public:
    using Action = TicTacToeAction;
    using Status = TicTacToeStatus;
    using BoardStateType = Eigen::Matrix<char, 3, 3>;

    //TODO: implement TicTacToe(string state) ... initialize x_turn based on num_x-num_o

    TicTacToe() {
        Reset();
    };

    BoardStateType GetBoardState() const {
        return board_state_;
    }

    void Reset() {
        game_status_ = TicTacToeStatus::IN_PROGRESS;
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
        game_status_ = UpdateTicTacToeStatus();
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

template <class Game, template <class Game> class Agent1, template <class Game> class Agent2>
class GameSession {
public:
    typename Game::Status PlayOnce() {
        while(true) {
            game.ApplyAction(player1.GetAction(game));
            if(game.GameOver()) {
                player2.GetAction(game);
                break;
            }
            game.ApplyAction(player2.GetAction(game));
            if(game.GameOver()) {
                player1.GetAction(game);
                break;
            }
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
        player1.Reset();
        player2.Reset();
    }

    Agent1<Game> GetPlayer1() const {
        return player1;
    }

    Agent2<Game> GetPlayer2() const {
        return player2;
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
    void Reset() { }
};

template <class Game>
class PickRandomActionAgent {

public:
    typename Game::Action GetAction(const Game& state) const {
        auto const& actions = state.GetAvailableActions();
        if(actions.size() > 0)
            return *select_randomly(actions.begin(), actions.end());
    }
    void Reset() { }
};


template <class Game>
class TreeSearchAgent {
public:
  struct GameResults {
    int wins, losses, draws;
    void print() {
      std::cout << "Wins: " << wins << std::endl;
      std::cout << "Losses: " << wins << std::endl;
      std::cout << "Draws: " << wins << std::endl;
    }
  };

  typename Game::Action GetAction(const Game& state) {
    if(game_tree.size() == 0) {
      using namespace std::chrono;
      high_resolution_clock::time_point t1 = high_resolution_clock::now();
      TreeSearch(state, true);
      high_resolution_clock::time_point t2 = high_resolution_clock::now();
      duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
      std::cout << "Building full game tree took " << time_span.count()*1000 << " ms." << std::endl;
    }

    typename Game::Action best_action;
    double best_proportion = 0;
    for(auto const& action : state.GetAvailableActions()) {      
      GameResults results; 
      Game result_of_action = state.ForwardModel(action);
      results = game_tree[result_of_action.GetStateString()];
      double proportion_of_wins = results.wins / (double)(results.wins + results.losses + results.draws);
      if(proportion_of_wins > best_proportion) {
        best_proportion = proportion_of_wins;
        best_action = action;
      }
    }
    return best_action;
  }
  
  GameResults TreeSearch(const Game& state, bool our_turn) { 
      GameResults results = {0, 0, 0};
      for(auto const& action : state.GetAvailableActions()) {
        Game result_of_action = state.ForwardModel(action);
        std::string state_string = result_of_action.GetStateString();
        if(result_of_action.GameOver()) { //action results in a terminal state
          if(!result_of_action.Draw()) { //its a win on our turn, loss otherwise
            if(our_turn) {
              game_tree[state_string] = {1, 0, 0};
              results.wins++;
            } else {
              game_tree[state_string] = {0, 1, 0};
              results.losses++;
            }
          } else { //its a draw
            game_tree[state_string] = {0, 0, 1};
            results.draws++;
          }
        } else {
          GameResults child_results = TreeSearch(result_of_action, !our_turn);
          results.wins += child_results.wins;
          results.losses += child_results.losses;
          results.draws += child_results.draws;
        }
      }
    game_tree[state.GetStateString()] = results;
    return results;
    }
    std::unordered_map<std::string, GameResults> game_tree;
    void Reset() { }
};


template <class Game>
class TemporalDifferenceAgent {
public:

    typename Game::Action Policy(const Game& state) {
        float best_value = 0;
        typename Game::Action best_action;
        std::string state_string;
        for(auto const& action : state.GetAvailableActions()) {
            Game next_state = state.ForwardModel(action);
            float state_value = GetValue(next_state);
            if(state_value > best_value) {
                state_string = next_state.GetStateString();
                best_value = state_value;
                best_action = action;
            }
        }
        float Vs = value_function[state_after_last_move];
        float Vs_prime = best_value;
        value_function[state_after_last_move] = Vs + alpha * (Vs_prime - Vs);
        state_after_last_move = state_string;
        return best_action;
    }

    typename Game::Action GetAction(const Game& state) {
        if(state_after_last_move.empty()) {
            state_after_last_move = state.GetStateString();
        }

        typename Game::Action action;
        if(state.GameOver()) {
            std::string state_string = state.GetStateString();
            float Vs = value_function[state_after_last_move];
            value_function[state_after_last_move] = Vs - alpha * Vs;
            return action;
        }
        auto actions = state.GetAvailableActions();

        float random = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        if(random < epsilon) {
            action = *select_randomly(actions.begin(), actions.end());
        } else {
            action = Policy(state);
        }
        //handle exploratory actions
        return action;
    }

    void Reset() {
        state_after_last_move = "";
    }

private:
    float GetValue(const Game& state) {
        if(state.GameOver() && !state.Draw()) {
            return 1.0f;
        }
        std::string state_string = state.GetStateString();
        if(value_function.find(state_string) == value_function.end()) {
            value_function[state_string] = 0.5;
        }
        return value_function[state_string];
    }


    const float alpha = 0.05; //learning rate
    const float epsilon = 0.05; //exploration rate
    std::string state_after_last_move;
    std::unordered_map<std::string, float> value_function;
};




int main(int argc, char* argv[])  {
    using namespace std::chrono;
    int x_wins=0, o_wins=0, draws=0;
    int num_games = 1e6; 
    //TicTacToe game;
    //TreeSearchAgent<TicTacToe> tree_search_agent;
    //tree_search_agent.GetAction(game);

    GameSession<TicTacToe, TreeSearchAgent, TemporalDifferenceAgent> session;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    for(int i = 0; i < num_games ; i++) {
        switch(session.PlayOnce()) {
        case TicTacToeStatus::X_WINS:
            x_wins++;
            break;
        case TicTacToeStatus::O_WINS:
            o_wins++;
            break;
        case TicTacToeStatus::DRAW:
            draws++;
            break;
        }
        session.Reset();
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Played " << num_games/time_span.count() << " games per second." << std::endl;
    std::cout << "x_wins: " << x_wins/(float)num_games*100 << std::endl;
    std::cout << "o_wins: " << o_wins/(float)num_games*100 << std::endl;
    std::cout << "draws: " << draws/(float)num_games*100 << std::endl;
}
