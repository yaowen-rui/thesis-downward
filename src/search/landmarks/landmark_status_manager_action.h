#ifndef LANDMARKS_LANDMARK_STATUS_MANAGER_ACTION_H
#define LANDMARKS_LANDMARK_STATUS_MANAGER_ACTION_H

#include "transformer.h"                 
#include "../per_state_bitset.h"

namespace landmarks {

class LandmarkStatusManagerAction {
    LandmarkGraphAction &lm_action_graph;

public:
    // Construct from ActionLM so we can (1) access the action landmark graph and (2) the task's operator count.
    explicit LandmarkStatusManagerAction(Transformer &translator);

    // Bitset accessors for the current state.
    BitsetView get_past_action_landmarks(const State &state);
    BitsetView get_future_action_landmarks(const State &state);
    ConstBitsetView get_past_action_landmarks(const State &state) const;
    ConstBitsetView get_future_action_landmarks(const State &state) const;

    // Initialization at the initial state:
    // - past(A) = false for all A (no actions fired yet)
    // - future(A) = true for all A (every action-LM still has to occur)
    void progress_initial_state(const State &initial_state);

    //i am not sure if we need progress goals
    void progress_goals(const State &ancestor_state, BitsetView &future);

    void progress(const State &parent_state, OperatorID applied_op_id, const State &state);

    // overload with raw operator index.
    void progress(const State &parent_state, int applied_op_index, const State &state);

private:
    Transformer &translater;

    PerStateBitset pastA;   // past(A) — achieved at least once on path so far
    PerStateBitset futureA; // future(A) — must still occur after current state

    // - natural ordering: predecessors of each action-LM node (by node id)
    std::vector<std::vector<size_t>> preds_of;
    // - operators -> action-LM ids containing that operator
    std::vector<std::vector<size_t>> op_to_actionLMs;

    size_t num_action_lms = 0;
};

} 

#endif
