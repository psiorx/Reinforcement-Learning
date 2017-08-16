#include <iostream> 

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
    int num_games = 100;

    GameSession<TicTacToe, 
    MonteCarloTreeSearchAgent, MinimaxAgent> session;

    Stopwatch sw;

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
