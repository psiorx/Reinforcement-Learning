#include <iostream> 

#include "GameSession.h"
#include "TicTacToe.h"
#include "PickRandomActionAgent.h"
#include "MinimaxAgent.h"
#include "TemporalDifferenceAgent.h"
#include "MonteCarloTreeSearchAgent.h"
#include "TestGame.h"
#include "Stopwatch.h"
#include "QLearningAgent.h"

#include <unordered_map>

int main(int argc, char* argv[])  {
    std::srand ( unsigned ( std::time(0) ) );
    
    int x_wins=0, o_wins=0, draws=0;
    int num_games = 100;
    QLearningAgent<TicTacToe> agent;
    TicTacToe game;
    std::unordered_map<std::string, float> value_function;
    MinimaxAgent<TicTacToe> god;
    TemporalDifferenceAgent<TicTacToe> agent1;
    TemporalDifferenceAgent<TicTacToe> agent2;
    GameSession<TicTacToe, TemporalDifferenceAgent, TemporalDifferenceAgent> 
    training_session(game, agent1, agent2);

    for(int i = 0; i < 100 ; i++) {
        std::cout << "Epoch " << i << std::endl;
        agent1.SetExplorationRate(1.0/(i+1));
        agent2.SetExplorationRate(1.0/(i+1));
        agent1.SetLearningRate(1.0/(i+1));
        agent2.SetLearningRate(1.0/(i+1));
        training_session.PlayN(10000);
    }

    Stopwatch sw;
    GameSession<TicTacToe, TemporalDifferenceAgent, MinimaxAgent> 
      session(game, agent1, god);
 
    agent1.SetExplorationRate(0);
    player2.SetIterationLimit(500);

    sw.Start();
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
    sw.Stop();

    std::cout << "Played " 
    << num_games/sw.ElapsedMillis()*1000 
    << " games per second." << std::endl;

    std::cout << "x_wins: " 
    << x_wins/(float)num_games*100 
    << std::endl;

    std::cout << "o_wins: " 
    << o_wins/(float)num_games*100 
    << std::endl;

    std::cout << "draws: " 
    << draws/(float)num_games*100 
    << std::endl;
}
