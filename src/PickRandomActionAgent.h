#pragma once

#include "utils.h"

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