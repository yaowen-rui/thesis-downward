#ifndef LANDMARKS_LANDMARK_STATUS_MANAGER_ACTION_H
#define LANDMARKS_LANDMARK_STATUS_MANAGER_ACTION_H

#include "landmark_status_manager.h"  // base manager
#include "landmark.h"                 // LandmarkType to detect action nodes
#include "landmark_graph.h"

namespace landmarks {

/*
 * Usage:
 *   LandmarkStatusManagerAction lm_status(graph, prog_goals, prog_gn, prog_r);
 *   lm_status.progress_initial_state(initial_state);
 *   ...
 *   lm_status.progress(parent_state, applied_op_id, succ_state);
 *   auto futureA = lm_status.get_future_action_landmarks(state);
 *
 * Notes:
 *   - We call the base class methods first (facts), then compute action bits.
 *   - “Action needed” is defined via GN-children: action is future iff any
 *     GREEDY_NECESSARY child landmark is future (or initially not true).
 */
class LandmarkStatusManagerAction : public LandmarkStatusManager {
    LandmarkGraph &lm_graph;

    // Parallel bitsets for ACTION landmarks (indexed by LM node id)
    PerStateBitset past_action_landmarks;
    PerStateBitset future_action_landmarks;

    static bool is_action_node(const LandmarkNode *n);

public:
    LandmarkStatusManagerAction(
        LandmarkGraph &graph,
        bool progress_goals,
        bool progress_greedy_necessary_orderings,
        bool progress_reasonable_orderings);

    // Action bitset accessors
    BitsetView get_past_action_landmarks(const State &state);
    BitsetView get_future_action_landmarks(const State &state);
    ConstBitsetView get_past_action_landmarks(const State &state) const;
    ConstBitsetView get_future_action_landmarks(const State &state) const;

    // Initialize base (facts) then initialize action bitsets
    void progress_initial_state(const State &initial_state);

    // Progress base (facts) then progress action bitsets
    void progress(const State &parent_ancestor_state, OperatorID op_id,
                  const State &ancestor_state);
};

} 

#endif
