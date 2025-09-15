#include "landmark_status_manager_action.h"
#include <algorithm>
#include <cstddef>


using namespace std;

namespace landmarks {
LandmarkStatusManagerAction::LandmarkStatusManagerAction(ActionLM &translator)
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
        const LandmarkNodeAction *node = up.get();
        const size_t id = static_cast<size_t>(node->get_id());
        for (int op_id : node->get_landmarkAction().actions) {
            if (op_id >= 0 && op_id < static_cast<int>(op_to_actionLMs.size()))
                op_to_actionLMs[op_id].push_back(id);
        }
    }

    // Cache predecessor sets from the ordering in the action-LM graph.
    preds_of.assign(num_action_lms, {});
    for (const auto &up : lm_action_graph.get_nodes()) {
        const LandmarkNodeAction *node = up.get();
        const size_t v = static_cast<size_t>(node->get_id());
        for (const auto &pr : node->parents) {
            const LandmarkNodeAction *pnode = pr.first;
            preds_of[v].push_back(static_cast<size_t>(pnode->get_id()));
        }
    }
}

BitsetView LandmarkStatusManagerAction::get_past_action_landmarks(const State &state) {
    return pastA[state];
}
ConstBitsetView LandmarkStatusManagerAction::get_past_action_landmarks(const State &state) const {
    return pastA[state];
}

BitsetView LandmarkStatusManagerAction::get_future_action_landmarks(const State &state) {
    return futureA[state];
}
ConstBitsetView LandmarkStatusManagerAction::get_future_action_landmarks(const State &state) const {
    return futureA[state];
}

void LandmarkStatusManagerAction::progress_initial_state(const State &initial_state) {
    // past = 0, future = 1  
    BitsetView past   = get_past_action_landmarks(initial_state);
    BitsetView future = get_future_action_landmarks(initial_state);
    past.reset();
    future.reset();
    for (size_t i = 0; i < num_action_lms; ++i)
        future.set(i);
}

void LandmarkStatusManagerAction::progress_goals(const State &ancestor_state, BitsetView &future_out) {
    // Make 'future' consistent with current 'past' and ordering.
    ConstBitsetView past = get_past_action_landmarks(ancestor_state);
    // If caller passes 'future_out' as our internal future bitset view for this state,
    // this will directly update it. Otherwise it just writes into the provided view.
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
    BitsetView      past1   = get_past_action_landmarks(state);
    BitsetView      future1 = get_future_action_landmarks(state);

    // Copy parent flags to child, we'll update delta below.
    past1.reset();
    future1.reset();
    for (size_t i = 0; i < num_action_lms; ++i) {
        if (past0.test(i))   past1.set(i);
        if (future0.test(i)) future1.set(i);
    }

    // Which A contain the applied operator 'a'?
    const int a = applied_op_id.get_index();
    std::vector<char> affected(num_action_lms, 0);
    if (a >= 0 && a < static_cast<int>(op_to_actionLMs.size())) {
        for (size_t A : op_to_actionLMs[a]) {
            affected[A] = 1;

            // Rule: if a ∈ A then past(A) := true
            past1.set(A);

            // Rule: future(A) := false iff all predecessors of A were past in s0, else true
            bool all_pred_past_in_s0 = true;
            for (size_t P : preds_of[A]) {
                if (!past0.test(P)) { all_pred_past_in_s0 = false; break; }
            }
            if (all_pred_past_in_s0)
                future1.reset(A);
            else
                future1.set(A);
        }
    }

    // Handle all A with a ∉ A (your “else” branch).
    for (size_t A = 0; A < num_action_lms; ++A) {
        if (affected[A])
            continue;

        if (future0.test(A)) {
            // If future(A) was already true in s0, keep it true in s1.
            // (We already copied it; nothing to do.)
            continue;
        } else {
            // If future(A) was false in s0 but some predecessor of A is not past in s1, set future(A) true.
            bool some_pred_not_past_in_s1 = false;
            for (size_t P : preds_of[A]) {
                if (!past1.test(P)) { some_pred_not_past_in_s1 = true; break; }
            }
            if (some_pred_not_past_in_s1)
                future1.set(A);
            // Otherwise leave future(A) false and past(A) unchanged (already copied).
        }
    }
}

void LandmarkStatusManagerAction::progress(
    const State &parent_state, int applied_op_index, const State &state) {
    progress(parent_state, OperatorID(applied_op_index), state);
}

} 
