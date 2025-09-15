#ifndef ACTION_LM_H
#define ACTION_LM_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <memory>

#include "../task_proxy.h"
#include "../utils/hash.h"
#include "landmark_graph.h"
#include "landmark_factory.h"

namespace landmarks {
class AbstractTask;
class LandmarkGraph;
class LandmarkNode;
class LandmarkFactory;   

class LandmarkAction {
public:
    std::vector<int> actions; //operator IDs
    LandmarkAction(std::vector<int> op_IDs): actions(move(op_IDs)) {}

};

class LandmarkNodeAction {
    int id;
    LandmarkAction lm_action;
public:
    LandmarkNodeAction(LandmarkAction &&lm_action)
        : id(-1), lm_action(std::move(lm_action)) {
    }
    //for natural or weaker ordering
    std::unordered_map<LandmarkNodeAction *, EdgeType> parents;
    std::unordered_map<LandmarkNodeAction *, EdgeType> children;

    int get_id() const {return id;}

    void set_id(int new_id) {
        assert(id == -1 || new_id == id);
        id = new_id;
    }

    const LandmarkAction &get_landmarkAction() const {return lm_action;}

};

class LandmarkGraphAction {

public:
    LandmarkGraphAction() = default;;
    using Nodes = std::vector<std::unique_ptr<LandmarkNodeAction>>;

    LandmarkNodeAction &add_action_lm(LandmarkAction &&action_lm);
    void set_action_lm_ids();
    int get_num_action_lms() const {return nodes.size();}

    // Return all action-LM nodes that include this op_id.
    const std::vector<LandmarkNodeAction*>& get_action_lms(int op_id) const;
    const Nodes &get_nodes() const {return nodes;}

private:
    //op_id -> all action landmark nodes that contain this op.
    utils::HashMap<int, std::vector<LandmarkNodeAction*>> action_landmarks_to_nodes;

    Nodes nodes;
    
};
/*
 * ActionLM translates fact/disjunctive fact landmarks into action landmarks
 */
class ActionLM {
public:
    ActionLM(const TaskProxy &task_proxy,
            const std::shared_ptr<LandmarkFactory> &lm_factory,
            const std::shared_ptr<AbstractTask> &task);
    
    //map from fact lm node to action lm node(1:1)
    std::unordered_map<const LandmarkNode *, LandmarkNodeAction *> factNode_to_actionNode;

    LandmarkGraphAction action_lm_graph;

    std::shared_ptr<LandmarkFactory> lm_factory;//initialize in constructor, lm_factory should be passed in as a parameter
    utils::LogProxy log = utils::LogProxy(std::make_shared<utils::Log>("action_LM"));
    
    //build action lm graph (nodes+edges) from fact lm graph
    void build_action_lm_graph(LandmarkGraph *lm_graph);

    // per-node action set (nullptr if unknown)
    const std::vector<int> *get_action_achievers(const LandmarkNode *node) const;
    
    const TaskProxy &get_task_proxy() const { return task_proxy; }

    void compute_min_costs();// min cost per action-LM node (indexed by node id)
    
    void discard_all_orderings();

    int get_min_cost_per_action_lm(const LandmarkNodeAction &actionNode) const {
        return min_cost[actionNode.get_id()];
    }
    

private:
    std::shared_ptr<LandmarkGraph> lm_graph;
    void compute_lm_graph(const std::shared_ptr<AbstractTask> &task);//let landmarkFactory to comute_lm_graph
    std::vector<int> to_sorted_vector(std::unordered_set<int> &&s);//turn a set<int> into a sorted vector

    const TaskProxy task_proxy;
    
    std::unordered_map<int, int> bank_cost;//store all ops from action LM graph and its cost: op_id -> operator cost
    std::vector<int> min_cost;//min cost for each action LM in lm_action_graph (indexed by action_node_id)

    void setup_costs();

    //if there is order between two fact lm nodes, then add same order between their action lm nodes
    void edge_add(LandmarkNodeAction &from, LandmarkNodeAction &to, EdgeType type= EdgeType::NATURAL);
    void setUp_edge();
    
    
};

}

#endif
