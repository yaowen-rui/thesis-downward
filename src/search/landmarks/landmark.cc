#include "landmark.h"

using namespace std;

namespace landmarks {
bool Landmark::is_true_in_state(const State &state) const {
    //action landmarks don't correspind to a state, they are handled by the status manager; 
    //consider them never "true in state"
    if(type == LandmarkType::DISJ_ACTION) {
        return false;
    }
    
    if (disjunctive) {
        for (const FactPair &fact : facts) {
            if (state[fact.var].get_value() == fact.value) {
                return true;
            }
        }
        return false;
    } else {
        // conjunctive or simple
        for (const FactPair &fact : facts) {
            if (state[fact.var].get_value() != fact.value) {
                return false;
            }
        }
        return true;
    }
}

static Landmark make_disj_action(std::vector<int> &&ops) {
    Landmark lm(std::vector<FactPair>{}, /*disjunctive*/ false, /*conjunctive*/ false,
                /*is_true_in_goal*/ false, /*is_derived*/ false);
    lm.type = LandmarkType::DISJ_ACTION;
    lm.action_ids = std::move(ops);
    // expose the action_ids, so heuristics work unchanged.
    for (int id : lm.action_ids)
    {
        lm.first_achievers.insert(id);
        lm.possible_achievers.insert(id);
    }
    return lm;
}



}
