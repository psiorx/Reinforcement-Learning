#pragma once

template <class Game>
class TemporalDifferenceAgent {
public:
    TemporalDifferenceAgent(std::unordered_map<std::string, float>* value_function)
     : value_function(value_function) { }

    TemporalDifferenceAgent()
     : value_function(new std::unordered_map<std::string, float>()) { }

    void Experience(const std::string &state, 
                    const typename Game::Action& action, 
                    float reward, 
                    const std::string &next_state,
                    bool terminal) {

        float state_value = GetValue(state);
        float next_state_value = 0;

        if (!terminal) {
            next_state_value = GetValue(next_state);
        }

        float td_target = reward + next_state_value;
        (*value_function)[state] = state_value + alpha * (td_target - state_value);

        // std::cout << "------- TD Experience ------" << std::endl;
        // std::cout << "State: " << state << std::endl;
        // std::cout << "Reward: " << reward << std::endl;
        // std::cout << "Next State: " << next_state << std::endl;
        // std::cout << "Terminal: " << (int)terminal << std::endl;
        // std::cout << "state_value: " << state_value 
        //           << " next_state_value: " << next_state_value << std::endl;        
    }

    typename Game::Action GreedyAction(const Game& state, const std::vector<typename Game::Action>& actions) {
        float best_value = -100.0f;
        typename Game::Action best_action;
        for(auto const& action : actions) {
            Game next_state = state.ForwardModel(action);
            float state_value = value_sign * GetValue(next_state.GetStateString());
            if(state_value >= best_value) {
                best_value = state_value;
                best_action = action;
            }
        }
        return best_action;
    }

    typename Game::Action GetAction(const Game& state) {
        typename Game::Action action;

        auto actions = state.GetAvailableActions();

        float random = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        if(random < epsilon) {
            action = *select_randomly(actions.begin(), actions.end());
        } else {
            action = GreedyAction(state, actions);
        }
        
        return action;
    }

    void SetLearningRate(float alpha) {
        this->alpha = alpha;
    }

    void SetExplorationRate(float epsilon) {
        this->epsilon = epsilon;
    }

    void Reset() {

    }

    void Maximize() {
        value_sign = 1.0;
    }

    void Minimize() {
        value_sign = -1.0;
    }


private:
    float GetValue(const std::string &state_string) {
        if(value_function->find(state_string) == value_function->end()) {
            (*value_function)[state_string] = 0.0;
            return 0.0;
        }
        return (*value_function)[state_string];
    }

    float value_sign = 1.0;
    float alpha = 0.05; //learning rate
    float epsilon = 0.05; //exploration rate
    std::unordered_map<std::string, float>* value_function;
};