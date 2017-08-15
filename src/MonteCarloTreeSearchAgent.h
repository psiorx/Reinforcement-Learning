#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include "utils.h"
#include "Stopwatch.h"

struct GameStats {
  int wins;
  int plays;
};

template <class Game>
struct TreeNode {
  TreeNode(Game const& game, 
           bool our_turn, 
           std::shared_ptr<TreeNode<Game>> parent,
           typename Game::Action action)
    : stats({0, 0}), state(game), our_turn(our_turn), parent(parent), action(action) { 
        unexplored_actions = state.GetAvailableActions();
        std::random_shuffle(unexplored_actions.begin(), unexplored_actions.end());

        if(parent) {
        }
  }

  double WinRatio() const {
    return stats.wins / (double) stats.plays;
  }

  double UpperConfidenceBound() const {
  double win_percentage = WinRatio();
    
    if(!parent) {
      return win_percentage;
    }
    double confidence_bound = sqrt(2 * log(parent->stats.plays * 1.0) / stats.plays);
    return win_percentage + confidence_bound; 
  }

  GameStats stats;
  Game state;
  bool our_turn;
  typename Game::Action action;
  std::vector<std::shared_ptr<TreeNode<Game>>> children;
  std::vector<typename Game::Action> unexplored_actions;
  std::shared_ptr<TreeNode<Game>> parent;
};

template <class Game>
class MonteCarloTreeSearchAgent {
public:
  using TreeNodePtr = std::shared_ptr<TreeNode<Game>>;
  
  typename Game::Action GetAction(const Game& state) {
   search_tree = std::make_shared<TreeNode<Game>>(state, true, nullptr, typename Game::Action());
   //SearchForTime(100.0);
   
   SearchForIterations(10000);

   int best_value = 0;
   typename Game::Action best_action;
   for(auto const& child : search_tree->children) {
     double value = child->stats.plays;
     if(value >= best_value) {
       best_value = value;
       best_action = child->action;
     }
   }

   return best_action;
  }

  void SearchForTime(double ms) {
   const int batch_size = 1000;
   Stopwatch sw;
   sw.Start();
   int iterations = 0;
   while(sw.ElapsedMillis() < ms) {
    SearchForIterations(batch_size);
    iterations += batch_size;
   }
   sw.Stop();
  }

  void SearchForIterations(int n) {
    for(int i = 0; i < n; i++) {
       MonteCarloTreeSearch(search_tree);
     }   
  }

  TreeNodePtr Selection(TreeNodePtr node) {
    if(node == nullptr) {
      return node;
    }
  
    //check for unexplored actions
    size_t num_unexplored_actions = node->unexplored_actions.size();
    if(num_unexplored_actions != 0) {
      return node;
    }

    //treat as bandit problem
    double best_value = -std::numeric_limits<double>::infinity();
    TreeNodePtr best_child = nullptr;
    for(auto const& child : node->children) {
      double ucb = child->UpperConfidenceBound();
      if(ucb > best_value) {
        best_value = ucb;
        best_child = child;
      }
    }
    
    if(!best_child) {
     int score = GetScore(node);
     Backpropagation(node, score);
     return nullptr;
    }
     
    return Selection(best_child); 
  }

  TreeNodePtr Expansion(TreeNodePtr node) {
    auto action = node->unexplored_actions.back();
    node->unexplored_actions.pop_back();
    Game next_state = node->state.ForwardModel(action);
    auto child_node = std::make_shared<TreeNode<Game>>(next_state, !node->our_turn, node, action);
    node->children.push_back(child_node);
    return child_node;
  }

  int Simulation(TreeNodePtr node) {
    Game simulated_game = node->state;
    bool our_turn = node->our_turn;
    

    while(!simulated_game.GameOver()) {
      auto actions = simulated_game.GetAvailableActions();
      simulated_game.ApplyAction(*select_randomly(actions.begin(), actions.end()));
      our_turn = !our_turn;
    }
     

    int score;
    
    if(simulated_game.Draw()) {
      score = 0;
    } else if(our_turn) {
      score = -1;
    } else {
      score = 1;
    }

    return score;
  }

  int GetScore(TreeNodePtr const& node) {
    int score;
    if(node->state.Draw()) {
      score = 0;
    } else if(node->our_turn) {
      score = -1; 
    } else {
      score = 1;
    }

    return score;
  }

  void Backpropagation(TreeNodePtr node, int score) {

    node->stats.plays++;
    if(node->our_turn && score > 0) {
      node->stats.wins++;
    } else if(!node->our_turn && score < 0) {
      node->stats.wins--;
    }

    if(node->parent) {
      Backpropagation(node->parent, score);
    }
  }

  GameStats MonteCarloTreeSearch(TreeNodePtr node) {
    TreeNodePtr unexpanded_child = Selection(node);
    if(unexpanded_child) {
      TreeNodePtr expanded_node = Expansion(unexpanded_child);
      double reward = Simulation(expanded_node);
      Backpropagation(expanded_node, reward);
    } 
  }

  void Reset() { }

private:
  TreeNodePtr search_tree;
  std::unordered_map<std::string, TreeNodePtr> node_map;
};
