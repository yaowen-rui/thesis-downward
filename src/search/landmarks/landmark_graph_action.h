
#ifndef LANDMARKS_LANDMARK_GRAPH_ACTION_H
#define LANDMARKS_LANDMARK_GRAPH_ACTION_H

#include "landmark_graph.h"
#include "../utils/hash.h"

namespace landmarks {
    /*Inherits from landmark graph
    add a method to create an action landmark node. It does not kee its own index
    the factory is responsible for deduplication.
    Usage:
    LandmarkNode &actionNode = ag->add_action_landmark(ops);
    edge_add(actionNode, disj_fact_node, EdgeType::GREEDY_NECESSARY);
    */ 
class LandmarkGraphAction : public LandmarkGraph {

public:
    LandmarkGraphAction() = default;

    //create an action lm node from a set of operatior IDs
    LandmarkNode &add_action_landmark(std::vector<int> ops);
    //helper: identify if a node is an action lm node
    static bool is_action_node(const LandmarkNode *n);
};

} 

#endif
