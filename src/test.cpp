#include <iostream>
#include <unordered_map>

#include "GameSession.h"
#include "TicTacToe.h"
#include "PickRandomActionAgent.h"
#include "MinimaxAgent.h"
#include "TemporalDifferenceAgent.h"
#include "MonteCarloTreeSearchAgent.h"
#include "TestGame.h"
#include "Stopwatch.h"

int main(int argc, char* argv[])  {
    std::srand ( unsigned ( std::time(0) ) );

    int x_wins=0, o_wins=0, draws=0;
    int num_games = 1000;
    TicTacToe game;
    std::unordered_map<std::string, float> value_function, terminal_values;
    TemporalDifferenceAgent<TicTacToe> agent1(&value_function, &terminal_values);
    TemporalDifferenceAgent<TicTacToe> agent2(&value_function, &terminal_values);
    GameSession<TicTacToe, TemporalDifferenceAgent, TemporalDifferenceAgent> 
    training_session(game, agent1, agent2);

    int training_games = 50000;

    agent1.SetExplorationRate(1.0);
    agent2.SetExplorationRate(1.0);

    agent1.SetLearningRate(1);
    agent2.SetLearningRate(1);

    training_session.PlayN(training_games);
  
    Stopwatch sw;
    MinimaxAgent<TicTacToe> god;
    MonteCarloTreeSearchAgent<TicTacToe> mcts;
    GameSession<TicTacToe, MinimaxAgent, MonteCarloTreeSearchAgent>
    session(game, god, mcts);

   agent1.SetExplorationRate(0);
   agent2.SetExplorationRate(0);

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
