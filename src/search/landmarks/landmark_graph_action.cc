#include "landmark_graph_action.h"

#include <algorithm>
#include <utility>

namespace landmarks {
LandmarkNode &LandmarkGraphAction::add_action_landmark(std::vector<int> ops) {
    std::sort(ops.begin(), ops.end());
    ops.erase(std::unique(ops.begin(), ops.end()), ops.end());

    Landmark lm = Landmark::make_disj_action(std::move(ops));
    LandmarkNode &node = LandmarkGraph::add_landmark(std::move(lm));//should update LandmarkGraph::add_landmark
    return node;
}

bool LandmarkGraphAction::is_action_node(const LandmarkNode *n) {
    if (!n) return false;
    const Landmark &lm = n->get_landmark();

    return lm.type == LandmarkType::DISJ_ACTION;
}

}