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

    int training_epochs = 100;
    for(int i = 0; i < training_epochs ; i++) {
        std::cout << "Epoch " << i << std::endl;
        float decay = (training_epochs - i) / (float)training_epochs;
        agent1.SetExplorationRate(decay);
        agent2.SetExplorationRate(decay);
        agent1.SetLearningRate(decay);
        agent2.SetLearningRate(decay);
        training_session.PlayN(1000);
    }


    Stopwatch sw;
    MonteCarloTreeSearchAgent<TicTacToe> mcts;
    GameSession<TicTacToe, TemporalDifferenceAgent, MonteCarloTreeSearchAgent> 
      session(game, agent1, mcts);
 
    agent1.SetExplorationRate(0);
    // agent1.SetLearningRate(0);

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
