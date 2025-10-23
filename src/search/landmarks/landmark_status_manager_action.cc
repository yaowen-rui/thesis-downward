#include "landmark_status_manager_action.h"
#include <algorithm>
#include <cstddef>


using namespace std;

namespace landmarks {
LandmarkStatusManagerAction::LandmarkStatusManagerAction(Transformer &translator)
    : lm_action_graph(translator.action_lm_graph),
      translater(translator),
      pastA(vector<bool>(lm_action_graph.get_num_action_lms(), false)),
      futureA(vector<bool>(lm_action_graph.get_num_action_lms(), true)) {
    
        // Number of action landmarks (ids are already assigned by ActionLM).
    num_action_lms = lm_action_graph.get_num_action_lms();

    // Build op -> action-LM incidence  (uses ActionLM graph actions).
    const auto &ops = translater.get_task_proxy().get_operators();
    op_to_actionLMs.assign(ops.size(), {});
    for (const auto &up : lm_action_graph.get_nodes()) {
        const ActionLandmarkNode *node = up.get();
        const size_t id = static_cast<size_t>(node->get_id());
        for (int op_id : node->get_Actionlandmark().actions) {
            if (op_id >= 0 && op_id < static_cast<int>(op_to_actionLMs.size()))
                op_to_actionLMs[op_id].push_back(id);
        }
    }

    // Cache predecessor sets from the ordering in the action-LM graph.
    preds_of.assign(num_action_lms, {});
    for (const auto &up : lm_action_graph.get_nodes()) {
        const ActionLandmarkNode *node = up.get();
        const size_t v = static_cast<size_t>(node->get_id());
        for (const auto &pr : node->parents) {
            const ActionLandmarkNode *pnode = pr.first;
            preds_of[v].push_back(static_cast<size_t>(pnode->get_id()));
        }
    }
}

BitsetView LandmarkStatusManagerAction::get_past_action_landmarks(const State &state) {
    return pastA[state];
}
ConstBitsetView LandmarkStatusManagerAction::get_past_action_landmarks(const State &state) const {
    (void) const_cast<LandmarkStatusManagerAction*>(this)->pastA[state];
    return pastA[state];
}

BitsetView LandmarkStatusManagerAction::get_future_action_landmarks(const State &state) {
    return futureA[state];
}
ConstBitsetView LandmarkStatusManagerAction::get_future_action_landmarks(const State &state) const {
    (void) const_cast<LandmarkStatusManagerAction*>(this)->futureA[state];
    return futureA[state];
}

void LandmarkStatusManagerAction::progress_initial_state(const State &initial_state) {
    // For the initial state we explicitly set: past=false, future=true
    BitsetView past   = get_past_action_landmarks(initial_state);
    BitsetView future = get_future_action_landmarks(initial_state);
    for (size_t i = 0; i < num_action_lms; ++i) {
        past.reset(i);
        future.set(i);
    }
}

void LandmarkStatusManagerAction::progress_goals(const State &ancestor_state, BitsetView &future_out) {
    // Make 'future' consistent with current 'past' and natural ordering:
    // If A is not past or some predecessor of A is not past, keep A future; else clear A from future.
    ConstBitsetView past = get_past_action_landmarks(ancestor_state);
    for (size_t a = 0; a < num_action_lms; ++a) {
        bool some_pred_not_past = false;
        for (size_t p : preds_of[a]) {
            if (!past.test(p)) { some_pred_not_past = true; break; }
        }
        if (!past.test(a) || some_pred_not_past)
            future_out.set(a);
        else
            future_out.reset(a);
    }
}

void LandmarkStatusManagerAction::progress(
    const State &parent_state, OperatorID applied_op_id, const State &state) {
    if (state == parent_state)
        return;

    // Views of parent and current.
    ConstBitsetView past0   = get_past_action_landmarks(parent_state);
    ConstBitsetView future0 = get_future_action_landmarks(parent_state);

    const int op = applied_op_id.get_index();
    
    // ---- Per-edge progression result (basic rules) ----
    // Start from the parent's LM state for this edge-result.
    std::vector<char> past_edge(num_action_lms, 0);
    std::vector<char> future_edge(num_action_lms, 0);
    for (size_t i = 0; i < num_action_lms; ++i) {
        past_edge[i]   = past0.test(i) ? 1 : 0;
        future_edge[i] = future0.test(i) ? 1 : 0;
    }


    // ---- Natural-ordering pruning (edge-level conflict test) ----
    // If applied action 'a' is in some B and any predecessor A of B is not past in s,
    // this edge violates A ->_n B and must be excluded from merging.
    
    bool violates = false;
    if (op >= 0 && op < static_cast<int>(op_to_actionLMs.size())) {
        for (size_t B : op_to_actionLMs[op]) {
            for (size_t A : preds_of[B]) {
                if (!past0.test(A)) {
                    // Conflict: do not merge this edge into 'state'.
                    violates = true;
                    break;
                }
            }
            if (violates) break;
        }
    }
    
    // Apply Hit(op) if no violation: all action-LMs containing 'op' are achieved on this transition.
    if (!violates && op >= 0 && op < static_cast<int>(op_to_actionLMs.size())) {
        for (size_t A : op_to_actionLMs[op]) {
            past_edge[A] = 1;   // A enters past for this edge
            future_edge[A] = 0; // A leaves future for this edge
        }
    }

    // ---- Merge this edge-result into the target state's stored bitsets ----
    //   Merge across multiple incoming paths to the same StateID:
    BitsetView past1   = get_past_action_landmarks(state);
    BitsetView future1 = get_future_action_landmarks(state);

    for (size_t i = 0; i < num_action_lms; ++i) {
        // // Intersection for past
        // if (!past_edge[i] && past1.test(i)) {
        //     past1.reset(i);
        // }
        // // Union for future
        // if (future_edge[i] && !future1.test(i)) {
        //     future1.set(i);
        // }
        //OR for past
       if (past_edge[i]) {
            past1.set(i);
        }
        // AND for future
        if (!future_edge[i] && future1.test(i)) {
            future1.reset(i);
        }
    }
}

void LandmarkStatusManagerAction::progress(
    const State &parent_state, int applied_op_index, const State &state) {
    progress(parent_state, OperatorID(applied_op_index), state);
}

} 
