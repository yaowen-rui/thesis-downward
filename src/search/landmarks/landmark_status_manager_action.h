#ifndef LANDMARKS_LANDMARK_STATUS_MANAGER_ACTION_H
#define LANDMARKS_LANDMARK_STATUS_MANAGER_ACTION_H

#include "action_LM.h"                 
#include "../per_state_bitset.h"

namespace landmarks {

class LandmarkStatusManagerAction {
    LandmarkGraphAction &lm_action_graph;

public:
    // Construct from ActionLM so we can (1) access the action-LM graph and (2) the task's operator count.
    explicit LandmarkStatusManagerAction(ActionLM &translator);

    // Bitset accessors for the current state.
    BitsetView get_past_action_landmarks(const State &state);
    BitsetView get_future_action_landmarks(const State &state);
    ConstBitsetView get_past_action_landmarks(const State &state) const;
    ConstBitsetView get_future_action_landmarks(const State &state) const;

    // Initialization at the initial state:
    // - past(A) = false for all A (no actions fired yet)
    // - future(A) = true for all A (every action-LM still has to occur)
    void progress_initial_state(const State &initial_state);

    // (Optional) make 'future' consistent w.r.t. current 'past' in ancestor_state.
    // If some predecessor of A is not past, future(A) := true; else if past(A) then future(A) := false.
    void progress_goals(const State &ancestor_state, BitsetView &future);

    // rules:
    // If s0 --a--> s1 and a ∈ A:
    //   past(A) := true;
    //   future(A) := false iff all predecessors of A were past in s0; else true.
    // If a ∉ A:
    //   if future(A) was true in s0, keep it true;
    //   else if some predecessor of A is not past in s1, set future(A) := true;
    //   else leave future(A) false and past(A) unchanged.
    void progress(const State &parent_state, OperatorID applied_op_id, const State &state);

    // overload with raw operator index.
    void progress(const State &parent_state, int applied_op_index, const State &state);

private:
    ActionLM &translater;

    PerStateBitset pastA;   // past(A) — achieved at least once on path so far
    PerStateBitset futureA; // future(A) — must still occur after current state

    // - predecessors of each action-LM node (by node id)
    std::vector<std::vector<size_t>> preds_of;
    // - operators -> action-LM ids containing that operator
    std::vector<std::vector<size_t>> op_to_actionLMs;

    size_t num_action_lms = 0;
};

} 

#endif
