#include "landmark.h"

using namespace std;

namespace landmarks {
bool Landmark::is_true_in_state(const State &state) const {
    //action lms are not evaluated in states
    if (action_disjuncive) {
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
Landmark Landmark::make_action_disjunctive(const std::vector<int> &op_ids) {
    Landmark lm({}, false, false, false, false, true);//action_disjunctive is true
    lm.op_ids = op_ids;//copy
    //sort and de-duplicate 
    std::sort(lm.op_ids.begin(), lm.op_ids.end());
    lm.op_ids.erase(std::unique(lm.op_ids.begin(), lm.op_ids.end()), lm.op_ids.end());
    return lm;
}

}
