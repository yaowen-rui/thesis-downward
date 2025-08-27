#ifndef LANDMARKS_LANDMARK_H
#define LANDMARKS_LANDMARK_H

#include "../task_proxy.h"

#include <unordered_set>

namespace landmarks {
enum class LandmarkType {
    SIMPLE_FACT,
    DISJ_FACT,
    CONJ_FACT,
    DISJ_ACTION //set of operator IDS
};

class Landmark {
public:
    Landmark(std::vector<FactPair> _facts, bool disjunctive, bool conjunctive,
             bool is_true_in_goal = false, bool is_derived = false)
        : facts(move(_facts)), disjunctive(disjunctive), conjunctive(conjunctive),
          is_true_in_goal(is_true_in_goal), is_derived(is_derived),
          //new : derive type internally
            type(conjunctive? LandmarkType::CONJ_FACT
                            : (disjunctive? LandmarkType::DISJ_FACT
                                          : LandmarkType::SIMPLE_FACT)) {
        assert(!(conjunctive && disjunctive));
        assert((conjunctive && facts.size() > 1)
            || (disjunctive && facts.size() > 1) || facts.size() == 1); 
    }

    bool operator ==(const Landmark &other) const {
        return this == &other;
    }

    bool operator !=(const Landmark &other) const {
        return !(*this == other);
    }

    std::vector<FactPair> facts;
    bool disjunctive;
    bool conjunctive;
    bool is_true_in_goal;
    bool is_derived;

    std::unordered_set<int> first_achievers;
    std::unordered_set<int> possible_achievers;

    bool is_true_in_state(const State &state) const;

    //new added for disj action lm
    LandmarkType type = LandmarkType::SIMPLE_FACT;
    std::vector<int> action_ids;
    //factory method to create a disjunctive action landmark
    static Landmark make_disj_action(std::vector<int> &&ops);
    bool is_action() const {
        return type == LandmarkType::DISJ_ACTION;
    }

};
}
#endif
