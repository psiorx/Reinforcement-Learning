#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>

//     A      our turn
//   B   C    their turn
// D  E F  G  our turn
//    H       their turn

//D = LOSS
//F = LOSS
//G = LOSS
//H = WIN

enum class TestGameStatus {
    WIN,
    LOSS,
    DRAW,
    IN_PROGRESS
};

struct TestGameNode {
    std::string state;
    std::vector<TestGameNode> children;
    TestGameStatus status;
};

class TestGame {
public:
    using Action = int;
    using Status = TestGameStatus;
    using BoardStateType = std::string;

    TestGame() {
        Reset();
    };

    BoardStateType GetBoardState() const {
        return current_node_.state;
    }

    void Reset() {
        game_status_ = TestGameStatus::IN_PROGRESS;

        TestGameNode node_D;
        node_D.state = "D";
        node_D.status = TestGameStatus::LOSS;

        TestGameNode node_F;
        node_F.state = "F";
        node_F.status = TestGameStatus::LOSS;

        TestGameNode node_G;
        node_G.state = "G";
        node_G.status = TestGameStatus::LOSS;

        TestGameNode node_H;
        node_H.state = "H";
        node_H.status = TestGameStatus::WIN;

        TestGameNode node_E;
        node_E.state = "E";
        node_E.status = TestGameStatus::IN_PROGRESS;
        node_E.children.push_back(node_H);

        TestGameNode node_B;
        node_B.state = "B";
        node_B.status = TestGameStatus::IN_PROGRESS;
        node_B.children.push_back(node_D);
        node_B.children.push_back(node_E);

        TestGameNode node_C;
        node_C.state = "C";
        node_C.status = TestGameStatus::IN_PROGRESS;
        node_C.children.push_back(node_F);
        node_C.children.push_back(node_G);

        root_.children.clear();

        root_.state = "A";
        root_.status = TestGameStatus::IN_PROGRESS;
        root_.children.push_back(node_B);
        root_.children.push_back(node_C);

        current_node_ = root_;
    }

    std::vector<Action> GetAvailableActions() const {
        std::vector<Action> actions;
        if(GameOver()) {
          return actions;
        }

        for(int i = 0; i < current_node_.children.size(); i++) {
          actions.push_back(i);
        }
        return actions;
    }

    void ApplyAction(Action const& action) {
        std::cout << "trying to apply action " << action << " from state " << GetStateString() << std::endl;
        auto selected_child = current_node_.children[action];
        std::cout << "selected child is: " << selected_child.state << std::endl;
        current_node_ = selected_child;
        game_status_ = current_node_.status;
    }

    TestGame ForwardModel(Action const& action) const {
        TestGame new_board(*this);
        new_board.ApplyAction(action);
        return new_board;
    }

    TestGameStatus GetGameStatus() const {
        return game_status_;
    }

    bool GameOver() const {
        return game_status_ != TestGameStatus::IN_PROGRESS;
    }

    bool Win() const  {
        return game_status_ == TestGameStatus::WIN;
     }

    bool Draw() const {
        return game_status_ == TestGameStatus::DRAW;
    }

    std::string GetStateString() const {
        return current_node_.state;
    }

private:
    TestGameStatus game_status_;
    TestGameNode current_node_;
    TestGameNode root_;
};