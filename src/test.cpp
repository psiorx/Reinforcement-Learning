#include <iostream> 

#include "GameSession.h"
#include "ConnectFour.h"
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
    QLearningAgent<ConnectFour> agent;
    ConnectFour game;
    std::unordered_map<std::string, float> value_function;
    TemporalDifferenceAgent<ConnectFour> agent1(&value_function);
    TemporalDifferenceAgent<ConnectFour> agent2(&value_function);
    GameSession<ConnectFour, TemporalDifferenceAgent, TemporalDifferenceAgent> 
    training_session(game, agent1, agent2);

    int training_games = 0;

    agent1.SetExplorationRate(0.2);
    agent2.SetExplorationRate(0.2);

    agent1.SetLearningRate(1);
    agent2.SetLearningRate(1);

    for(int i = 0; i < training_games; i++) {
        training_session.PlayOnce();
    }

    auto file = fopen("/home/psior/debug_values.csv", "w");
    for(std::pair<std::string, double> pair: value_function) {
        fprintf(file, "%s,%0.1f\n", pair.first.c_str(), pair.second);
    }
    fclose(file);

   Stopwatch sw;
   MonteCarloTreeSearchAgent<ConnectFour> mcts;
   GameSession<ConnectFour, TemporalDifferenceAgent, MonteCarloTreeSearchAgent>
     session(game, agent1, mcts);

   agent1.SetExplorationRate(0);
   agent2.SetExplorationRate(0);
   agent1.SetLearningRate(0);
   agent2.SetLearningRate(0);


   sw.Start();
   for(int i = 0; i < num_games ; i++) {
       switch(session.PlayOnce()) {
       case ConnectFourStatus::X_WINS:
           x_wins++;
           break;
       case ConnectFourStatus::O_WINS:
           o_wins++;
           break;
       case ConnectFourStatus::DRAW:
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
