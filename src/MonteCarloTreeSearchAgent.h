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
  using TreeNodePtr = TreeNode<Game>*;
  
  TreeNode(Game const& game, 
           bool our_turn, 
           TreeNodePtr parent,
           typename Game::Action action,
           float exploration_rate)
    : stats({0, 0}), 
      state(game), 
      our_turn(our_turn), 
      parent(parent), 
      action(action), 
      exploration_rate(exploration_rate) { 
        unexplored_actions = state.GetAvailableActions();
        std::random_shuffle(unexplored_actions.begin(), unexplored_actions.end());
  }

  double WinRatio() const {
    return stats.wins / (double) stats.plays;
  }

  double UpperConfidenceBound() const {
  double win_percentage = WinRatio();
    
    if(!parent) {
      return win_percentage;
    }
    double confidence_bound = sqrt(exploration_rate * log(parent->stats.plays * 1.0) / stats.plays);
    return win_percentage + confidence_bound; 
  }

  GameStats stats;
  Game state;
  bool our_turn;
  typename Game::Action action;
  float exploration_rate;
  std::vector<TreeNodePtr> children;
  std::vector<typename Game::Action> unexplored_actions;
  TreeNodePtr parent;
};

template <class Game>
class MonteCarloTreeSearchAgent {
public:
  MonteCarloTreeSearchAgent() : exploration_rate(2), iteration_limit(100) { }  
  using TreeNodePtr = TreeNode<Game>*;

  typename Game::Action GetAction(const Game& state) {
   auto root_node = std::unique_ptr<TreeNode<Game>>(new TreeNode<Game>(state, true, nullptr, typename Game::Action(), exploration_rate));
   search_tree = root_node.get();
   nodes.clear();
   nodes.push_back(std::move(root_node));

   SearchForIterations(iteration_limit);

   double best_value = -10;
   typename Game::Action best_action;
   for(auto const& child : search_tree->children) {
     //std::cout << child->stats.wins << "/" << child->stats.plays << std::endl;
     //double value = child->WinRatio();
     double value = child->stats.plays;
     if(value >= best_value) {
       best_value = value;
       best_action = child->action;
     }
   }

   return best_action;
  }

  void Reset() { }
  void Update(const Game& new_state, float reward) { }

  void SetIterationLimit(size_t iterations) {
    iteration_limit = iterations;
  }

  void SetExplorationRate(float rate) {
    exploration_rate = rate;
  }

private:
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
     Backpropagation(node, node->WinRatio());
     return nullptr;
    }
     
    return Selection(best_child); 
  }

  TreeNodePtr Expansion(TreeNodePtr node) {
    
    if(node->state.GameOver()) {
      std::cout << "Tried to expand a terminal state" << std::endl;
    }

    auto action = node->unexplored_actions.back();
    node->unexplored_actions.pop_back();
    Game next_state = node->state.ForwardModel(action);
    
    auto child_node = std::unique_ptr<TreeNode<Game>>(new TreeNode<Game>(next_state, !node->our_turn, node, action, exploration_rate));
    node->children.push_back(child_node.get());
    nodes.push_back(std::move(child_node));
    return node->children.back();
  }

  int Simulation(TreeNodePtr node) {
    Game simulated_game = node->state;
    bool our_turn = node->our_turn;
    
    float reward;
    while(!simulated_game.GameOver()) {
      auto actions = simulated_game.GetAvailableActions();
      reward = simulated_game.ApplyAction(*select_randomly(actions.begin(), actions.end()));
      our_turn = !our_turn;
    }

    int score;
    if(simulated_game.Draw()) {
      score = 0;
    } else if(our_turn == node->our_turn) {
      score = reward;
    } else {
      score = -reward;
    }
    return score;
  }

  int GetScore(TreeNodePtr const& node) {
    int score;
    if(node->state.Draw()) {
      score = 0;
    } else {
      score = 1;
    }
    return score;
  }

  void Backpropagation(TreeNodePtr node, int score) {
    node->stats.plays++;
    node->stats.wins += score;
    if(node->parent) {
      Backpropagation(node->parent, -score);
    }
  }

  void MonteCarloTreeSearch(TreeNodePtr node) {
    TreeNodePtr unexpanded_child = Selection(node);

    if(unexpanded_child) {
      TreeNodePtr expanded_node = Expansion(unexpanded_child);
      double reward = Simulation(expanded_node);
      Backpropagation(expanded_node, reward);
    } 
  }


  size_t iteration_limit;
  float exploration_rate;
  std::vector<std::unique_ptr<TreeNode<Game>> > nodes;
  TreeNodePtr search_tree;
};
