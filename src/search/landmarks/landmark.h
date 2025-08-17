#ifndef LANDMARKS_LANDMARK_H
#define LANDMARKS_LANDMARK_H

#include "../task_proxy.h"

#include <unordered_set>
#include <vector>

namespace landmarks {
class Landmark {
public:
    Landmark(std::vector<FactPair> _facts, bool disjunctive, bool conjunctive,
             bool is_true_in_goal = false, bool is_derived = false, bool action_disjunctive = false)
        : facts(move(_facts)), disjunctive(disjunctive), conjunctive(conjunctive),
          is_true_in_goal(is_true_in_goal), is_derived(is_derived), action_disjuncive(action_disjunctive) {
        assert(!(conjunctive && disjunctive));
        if(!action_disjunctive){//assert checks only apply to fact-based landmarks
            assert((conjunctive && facts.size() > 1)
               || (disjunctive && facts.size() > 1) || facts.size() == 1);
        }
        
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

    bool action_disjuncive;
    bool is_action_disj() const { return action_disjuncive; }
    bool is_fact_disj()  const { return !action_disjuncive && disjunctive; }
    bool is_fact_conj()  const { return !action_disjuncive && conjunctive; }
    bool is_fact_based() const { return !action_disjuncive; }
    bool is_fact_simple() const {return !action_disjuncive && !disjunctive && !conjunctive && facts.size() == 1; }
    std::vector<int> op_ids;

    std::unordered_set<int> first_achievers;
    std::unordered_set<int> possible_achievers;

    bool is_true_in_state(const State &state) const;

    //build a disj action lm with the given operator IDs
    static Landmark make_action_disjunctive(const std::vector<int> &op_ids);
};
}
#endif
