#include "landmark_status_manager_action.h"
#include <algorithm> 

namespace landmarks {

static bool lm_is_true_in_state(const LandmarkNode *n, const State &s) {
    return n && n->get_landmark().is_true_in_state(s);
}

bool LandmarkStatusManagerAction::is_action_node(const LandmarkNode *n) {
    if (!n) return false;
    const Landmark &lm = n->get_landmark();
    return lm.type == LandmarkType::DISJ_ACTION;
}

LandmarkStatusManagerAction::LandmarkStatusManagerAction(
    LandmarkGraph &graph,
    bool progress_goals,
    bool progress_greedy_necessary_orderings,
    bool progress_reasonable_orderings)
  : LandmarkStatusManager(graph, progress_goals,
                          progress_greedy_necessary_orderings,
                          progress_reasonable_orderings),
    lm_graph(graph),
    // Initialize action bitsets: past=false, future=false by default
    past_action_landmarks(std::vector<bool>(graph.get_num_landmarks(), false)),
    future_action_landmarks(std::vector<bool>(graph.get_num_landmarks(), false)) {
    // Base ctor wiring, sizes and goal/reasonable/GN caches are handled there. :contentReference[oaicite:2]{index=2}
}

BitsetView LandmarkStatusManagerAction::get_past_action_landmarks(const State &state) {
    return past_action_landmarks[state];
}
BitsetView LandmarkStatusManagerAction::get_future_action_landmarks(const State &state) {
    return future_action_landmarks[state];
}
ConstBitsetView LandmarkStatusManagerAction::get_past_action_landmarks(const State &state) const {
    return past_action_landmarks[state];
}
ConstBitsetView LandmarkStatusManagerAction::get_future_action_landmarks(const State &state) const {
    return future_action_landmarks[state];
}

void LandmarkStatusManagerAction::progress_initial_state(const State &initial_state) {
    // 1) run base initialization for FACT landmarks (fills past/future) :contentReference[oaicite:3]{index=3}
    LandmarkStatusManager::progress_initial_state(initial_state);

    // 2) initialize ACTION landmark bitsets
    BitsetView pastA   = get_past_action_landmarks(initial_state);
    BitsetView futureA = get_future_action_landmarks(initial_state);

    pastA.reset();   // initially, no action was applied
    futureA.reset(); // set selectively below

    for (auto &node_up : lm_graph.get_nodes()) { // lm_graph comes from base class
        LandmarkNode *n = node_up.get();
        if (!is_action_node(n))
            continue;

        const int id = n->get_id();

        // "Needed initially" iff any GREEDY_NECESSARY child is not true initially
        bool needed = false;
        for (auto &ch : n->children) {
            if (ch.second == EdgeType::GREEDY_NECESSARY) {
                if (!lm_is_true_in_state(ch.first, initial_state)) {
                    needed = true;
                    break;
                }
            }
        }
        if (needed)
            futureA.set(id);
        else
            futureA.reset(id);
        // pastA stays reset(id)
    }
}

void LandmarkStatusManagerAction::progress(
    const State &parent_ancestor_state, OperatorID applied_op_id,
    const State &ancestor_state) {

    if (ancestor_state == parent_ancestor_state) {
        // Mirror base early-out for degenerate transitions. :contentReference[oaicite:4]{index=4}
        return;
    }

    // 1) progress FACT landmarks in base manager (updates past/future) :contentReference[oaicite:5]{index=5}
    LandmarkStatusManager::progress(parent_ancestor_state, applied_op_id, ancestor_state);

    // 2) progress ACTION landmark bitsets
    ConstBitsetView parent_pastA   = get_past_action_landmarks(parent_ancestor_state);
    BitsetView pastA               = get_past_action_landmarks(ancestor_state);
    BitsetView futureA             = get_future_action_landmarks(ancestor_state);

    // We also need current FACT 'future' to decide if an action is still needed
    ConstBitsetView futureFacts    = get_future_landmarks(ancestor_state);

    const int op_id = applied_op_id.get_index();

    // Inherit & update
    for (auto &node_up : lm_graph.get_nodes()) {
        LandmarkNode *n = node_up.get();
        if (!is_action_node(n))
            continue;

        const int id = n->get_id();
        const Landmark &alm = n->get_landmark();

        // ---- PAST(action): monotone + "consume if just applied"
        if (parent_pastA.test(id)) {
            pastA.set(id);
        } else {
            // Not past previously: check if the applied operator is one of the action IDs
            // (action_ids were canonicalized when node was created)
            if (std::binary_search(alm.action_ids.begin(), alm.action_ids.end(), op_id)) {
                pastA.set(id);
            } else {
                pastA.reset(id);
            }
        }

        // ---- FUTURE(action): needed iff any GN child is FUTURE (fact) now
        bool needed = false;
        for (auto &ch : n->children) {
            if (ch.second == EdgeType::GREEDY_NECESSARY) {
                if (futureFacts.test(ch.first->get_id())) {
                    needed = true;
                    break;
                }
            }
        }
        if (needed)
            futureA.set(id);
        else
            futureA.reset(id);
    }
}

}
