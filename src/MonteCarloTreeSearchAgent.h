#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include "utils.h"

struct GameStats {
  int wins;
  int plays;
};

template <class Game>
struct TreeNode {
  TreeNode(Game const& game, bool our_turn, std::shared_ptr<TreeNode<Game>> parent)
    : stats({0, 0}), state(game), our_turn(our_turn), parent(parent) { 
        unexplored_actions = state.GetAvailableActions();
        std::cout 
        << "Instantiated new TreeNode for state: " 
        << state.GetStateString() 
        << " with " 
        << state.GetAvailableActions().size() 
        << " unexplored actions" << std::endl;
        std::random_shuffle(unexplored_actions.begin(), unexplored_actions.end());
  }

  double UpperConfidenceBound() const {
    return stats.wins / (double) stats.plays + sqrt(2*log(parent->stats.plays) / stats.plays); 
  }

  GameStats stats;
  Game state;
  bool our_turn;
  std::vector<std::shared_ptr<TreeNode<Game>>> children;
  std::vector<typename Game::Action> unexplored_actions;
  std::shared_ptr<TreeNode<Game>> parent;
};

template <class Game>
class MonteCarloTreeSearchAgent {
public:

  using TreeNodePtr = std::shared_ptr<TreeNode<Game>>;
  
  typename Game::Action GetAction(const Game& state) {
    search_tree = std::make_shared<TreeNode<Game>>(state, true, nullptr);
    MonteCarloTreeSearch(search_tree);
  }

  TreeNodePtr Selection(TreeNodePtr node) {
    if(node == nullptr) {
      std::cout << "Selection phase ended with null" << std::endl;
      return node;
    }
  
    //check for unexplored actions
    if(node->unexplored_actions.size() != 0) {
      std::cout << "Ending selection phase" << std::endl;
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
    return Selection(best_child); 
  }

  TreeNodePtr Expansion(TreeNodePtr node) {
    auto action = node->unexplored_actions.back();
    node->unexplored_actions.pop_back();
    Game next_state = node->state.ForwardModel(action);
    auto child_node = std::make_shared<TreeNode<Game>>(next_state, !node->our_turn, node);
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
      std::cout << "Simulation ended in a draw" << std::endl;
      score = 0;
    } else if(our_turn) {
      std::cout << "Simulation ended in a loss" << std::endl;
      score = 0; 
    } else {
      std::cout << "Simulation ended in a win" << std::endl;
      score = 1;
    }
    std::cout << simulated_game.GetBoardState() << std::endl;
    return score;
  }

  void Backpropagation(TreeNodePtr node, int score) {
    if(!node) {
      return;
    }

    node->stats.plays++;
    node->stats.wins += score;
    std::cout << "New score for state :" << node->state.GetStateString() << ": " << node->stats.wins << "/" << node->stats.plays << std::endl;
    Backpropagation(node->parent, score);
  }

  GameStats MonteCarloTreeSearch(TreeNodePtr node) {
    std::cout << "Phase 1: Selection" << std::endl;
    TreeNodePtr unexpanded_child = Selection(node);
    if(unexpanded_child) {
      std::cout << "Phase 2: Expansion" << std::endl;
      TreeNodePtr expanded_node = Expansion(node);
      std::cout << "Phase 3: Simulation" << std::endl;
      double reward = Simulation(expanded_node);
      std::cout << "Phase 4: Backpropagation" << std::endl;
      Backpropagation(expanded_node, reward);
    }
  
    //Phase 4: Backpropagation
  }

  void Reset() { }

private:
  TreeNodePtr search_tree;
  std::unordered_map<std::string, TreeNodePtr> node_map;
};
