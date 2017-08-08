#pragma once

template <class Game>
class TemporalDifferenceAgent {
public:

    typename Game::Action GreedyAction(const Game& state, const std::vector<typename Game::Action>& actions) {
        float best_value = 0;
        typename Game::Action best_action;
        std::string state_string;
        for(auto const& action : actions) {
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
            action = GreedyAction(state, actions);
        }
        
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