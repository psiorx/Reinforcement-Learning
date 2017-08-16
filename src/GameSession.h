#pragma once
#include <vector>

template <class Game, template <class Game> class Agent1, template <class Game> class Agent2>
class GameSession {
public:
    GameSession() { }

    typename Game::Status PlayOnce() {
        while(true) {
            game.ApplyAction(player1.GetAction(game));
            if(game.GameOver()) {
                player2.GetAction(game);
                break;
            }
            auto test = player2.GetAction(game);
            game.ApplyAction(test);
            if(game.GameOver()) {
                player1.GetAction(game);
                break;
            }
        }
        return game.GetGameStatus();
    }

    std::vector<typename Game::Status> PlayN(std::size_t n) {
        std::vector<typename Game::Status> status_results;
        status_results.reserve(n);
        for(std::size_t count = 0; count < n ; count++) {
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