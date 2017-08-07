#pragma once

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