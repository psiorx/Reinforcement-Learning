#pragma once

template <class Game>
class QLearningAgent {
public:
    void Experience(const typename Game::BoardStateType& state, 
                    const typename Game::Action& action, 
                    float reward, 
                    const typename Game::BoardStateType& next_state,
                    bool terminal) {
        
    }

    typename Game::Action GreedyAction(const Game& state, const std::vector<typename Game::Action>& actions) {
        float best_value = -std::numeric_limits<float>::max();
        typename Game::Action best_action; //use ptr here?
        std::string state_string;
        for(auto const& action : actions) {
            float value = GetValue(state, action);
            if(value > best_value) {
                best_action = action;
                best_value = value;
            }
        }
        last_action = best_action;
        return best_action;
    }

    void Update(const Game& new_state, float reward) {
        float old_value = GetValue(state_after_last_move, last_action);
        auto state_action = std::make_pair(state_after_last_move, last_action);
        float best_value = -std::numeric_limits<float>::max();
        for(auto const& action : new_state.GetAvailableActions()) {
            float value = GetValue(new_state, action);
            if(value > best_value) {
                best_value = value;
            }
        }        
        //Q-Learning
        float learned_value = reward + discount_factor * best_value;
        value_function[state_action] = old_value + learning_rate * (learned_value - old_value);
        state_after_last_move = new_state;
 
    }


//     typename Game::Action GetAction(const Game& state) {
//         if(state_after_last_move.empty()) {
//             state_after_last_move = state.GetStateString();
//         }

//         typename Game::Action action;
//         if(state.GameOver()) {
//             std::string state_string = state.GetStateString();
//             float Vs = value_function[state_after_last_move];
//             value_function[state_after_last_move] = Vs - alpha * Vs;
//             return action;
//         }
//         auto actions = state.GetAvailableActions();

//         float random = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//         if(random < epsilon) {
//             action = *select_randomly(actions.begin(), actions.end());
//         } else {
//             action = GreedyAction(state, actions);
//         }
        
//         return action;
//     }

//     void SetLearningRate(float alpha) {
//         this->alpha = alpha;
//     }

//     void SetExplorationRate(float epsilon) {
//         this->epsilon = epsilon;
//     }

//     void Reset() {
//         state_after_last_move = "";
//     }

// private:

    float GetValue(const Game& state, typename Game::Action const& action) {
        if(state.GameOver()) {
            if(!state.Draw()) {
                return -1.0f; //1.0;
            }
        }
        std::string state_string = state.GetStateString();

        StateActionPair state_action = std::make_pair(state_string, action);

        if(value_function.find(state_action) == value_function.end()) {
            value_function[state_action] = 0.5;
        }
        return value_function[state_string];
    }

    float GetReward(const Game& state) {

    }


    float learning_rate = 0.05; //alpha
    float exploration_rate = 0.05; //epsilon
    float discount_factor = 1; //gamma

    std::string state_after_last_move;
    typename Game::Action last_action;

    struct pair_hash {
        template <typename T1, typename T2>
        std::size_t operator () (const std::pair<T1, T2>& p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            size_t res = 17;
            res = res * 31 + h1;
            res = res * 31 + h2;
            return res;
        }
    };

    using StateActionPair = std::pair<std::string, typename Game::Action>;
    std::unordered_map<StateActionPair, float, pair_hash> value_function;
};