#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <memory>
#include <cassert>
#include <utility>


#include "../task_proxy.h"
#include "../abstract_task.h"
#include "../utils/hash.h"
#include "landmark_graph.h"
#include "landmark_factory.h"
#include "../utils/logging.h"

namespace landmarks {

class LandmarkGraph;
class LandmarkNode;
class LandmarkFactory;   

class ActionLandmark {
public:
    std::vector<int> actions; //operator IDs
    ActionLandmark(std::vector<int> op_IDs): actions(std::move(op_IDs)) {}

};

class ActionLandmarkNode {
    int id;
    ActionLandmark lm_action;
public:
    ActionLandmarkNode(ActionLandmark &&lm_action)
        : id(-1), lm_action(std::move(lm_action)) {
    }
    //for natural or weaker ordering
    std::unordered_map<ActionLandmarkNode *, EdgeType> parents;
    std::unordered_map<ActionLandmarkNode *, EdgeType> children;

    int get_id() const {return id;}

    void set_id(int new_id) {
        assert(id == -1 || new_id == id);
        id = new_id;
    }

    const ActionLandmark &get_Actionlandmark() const {return lm_action;}

};

class LandmarkGraphAction {

public:
    LandmarkGraphAction() = default;;
    using Nodes = std::vector<std::unique_ptr<ActionLandmarkNode>>;

    ActionLandmarkNode &add_action_lm(ActionLandmark &&action_lm);
    void set_action_lm_ids();
    int get_num_action_lms() const {return nodes.size();}

    // Return all action-LM nodes that include this op_id.
    const std::vector<ActionLandmarkNode*>& get_action_lms(int op_id) const;
    const Nodes &get_nodes() const {return nodes;}

private:
    //op_id -> all action landmark nodes that contain this op.
    utils::HashMap<int, std::vector<ActionLandmarkNode*>> action_landmarks_to_nodes;

    Nodes nodes;
    
};
/*
 * ActionLM translates fact/disjunctive fact landmarks into action landmarks
 */
class Transformer {
public:
    Transformer(const TaskProxy &task_proxy,
            const std::shared_ptr<LandmarkFactory> &lm_factory,
            const std::shared_ptr<AbstractTask> &task);
    
    //map from fact lm node to action lm node(1:1)
    std::unordered_map<const LandmarkNode *, ActionLandmarkNode *> factNode_to_actionNode;

    LandmarkGraphAction action_lm_graph;

    std::shared_ptr<LandmarkFactory> lm_factory;//initialize in constructor, lm_factory should be passed in as a parameter
    utils::LogProxy log = lm_factory->log;

    
    //build action lm graph (nodes+edges) from fact lm graph
    void build_action_lm_graph(LandmarkGraph *lm_graph);

    // per-node action set (nullptr if unknown)
    const std::vector<int> *get_action_achievers(const LandmarkNode *node) const;
    
    const TaskProxy &get_task_proxy() const { return task_proxy; }

    void compute_min_costs();// min cost per action-LM node (indexed by node id)
    
    void discard_all_orderings();

    int get_min_cost_per_action_lm(const ActionLandmarkNode &actionNode) const {
        return min_cost[actionNode.get_id()];
    }
    //get min cost for each action LM in lm_action_graph (indexed by action_node_id)
    std::vector<int> get_min_cost() const {return min_cost;}
    

private:
    std::shared_ptr<LandmarkGraph> lm_graph;
    void compute_lm_graph(const std::shared_ptr<AbstractTask> &task);//let landmarkFactory to comute_lm_graph
    std::vector<int> to_sorted_vector(std::unordered_set<int> &&s);//turn a set<int> into a sorted vector

    const TaskProxy task_proxy;
    
    std::unordered_map<int, int> bank_cost;//store all ops from action LM graph and its cost: op_id -> operator cost
    std::vector<int> min_cost;//min cost for each action LM in lm_action_graph (indexed by action_node_id)

    void setup_costs();

    //if there is order between two fact lm nodes, then add same order between their action lm nodes
    void edge_add(ActionLandmarkNode &from, ActionLandmarkNode &to, EdgeType type= EdgeType::NATURAL);
    void setUp_edge();
    
    
};

}

#endif
