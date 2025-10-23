#include "transformer.h"

#include <algorithm>
#include <unordered_set>
#include <limits>
#include "util.h"

#include "../task_utils/task_properties.h"
#include "../utils/logging.h"
#include "landmark_graph.h"
#include "landmark_factory.h"

using namespace std;

namespace landmarks {
void LandmarkGraphAction::set_action_lm_ids() {
    int id = 0;
    for (auto &lmn : nodes) {
        lmn->set_id(id);
        ++id;
    }
}

ActionLandmarkNode &LandmarkGraphAction::add_action_lm(ActionLandmark &&action_lm) {
    unique_ptr<ActionLandmarkNode> new_node =
        utils::make_unique_ptr<ActionLandmarkNode>(move(action_lm));
    ActionLandmarkNode &new_node_ptr = *new_node;
    const ActionLandmark &lm = new_node->get_Actionlandmark();
    nodes.push_back(move(new_node));

    //avoid double-inserting the same op_id from this LM (if lm.actions has duplicates)
    std::unordered_set<int> seen;
    seen.reserve(lm.actions.size());
    for(int op_id: lm.actions) {
        if (seen.insert(op_id).second) {
            action_landmarks_to_nodes[op_id].push_back(&new_node_ptr);
        }
    }
    return new_node_ptr;
}

//use op_id to find all action-LM nodes that include this op_id.
const std::vector<ActionLandmarkNode*>& LandmarkGraphAction::get_action_lms(int op_id) const {
    static const std::vector<ActionLandmarkNode*> kEmpty;
    auto it = action_landmarks_to_nodes.find(op_id);
    return it == action_landmarks_to_nodes.end() ? kEmpty : it->second;
}


Transformer::Transformer(const TaskProxy &task_proxy,
                   const shared_ptr<LandmarkFactory> &factory,
                   const shared_ptr<AbstractTask> &task)
    : lm_factory(factory),
      task_proxy(task_proxy) {
    //1. build fact lm graph using lm factory
    compute_lm_graph(task);

    //2. buid action lm graph
    if(lm_graph) {
        build_action_lm_graph(lm_graph.get());
    }
    //3. cache operator costs 
    setup_costs();
    //4. min cost per action lm (aligned to action node id)
    compute_min_costs();
     
}

void Transformer::setup_costs() {
    bank_cost.clear();
    bank_cost.reserve(64);
    
    // Gather unique op ids from the action-LM graph
    std::unordered_set<int> all_ids;
    for (const auto &up : action_lm_graph.get_nodes()) {
        const auto &acts = up->get_Actionlandmark().actions;
        all_ids.insert(acts.begin(), acts.end());
    }

    auto ops_proxy = task_proxy.get_operators();
    for (int id : all_ids) {
        if (id >= 0 && id < static_cast<int>(ops_proxy.size())) {
            bank_cost[id] = ops_proxy[id].get_cost();
        }
    }
}

void Transformer::compute_lm_graph(const shared_ptr<AbstractTask> &task) {
    if (!lm_factory)
        return;
    lm_graph = lm_factory->compute_lm_graph(task);
}

//turn a set<int> into a sorted vector
vector<int> Transformer::to_sorted_vector(std::unordered_set<int> &&s) {
  vector<int> out(s.begin(), s.end());
  sort(out.begin(), out.end());
  out.erase(unique(out.begin(), out.end()), out.end());
  return out;
}

void Transformer::build_action_lm_graph(LandmarkGraph *lm_graph) {
    utils::Timer timer;
    auto &nodes = lm_graph->get_nodes();//vector<unique_ptr<LandmarkNode>>&
    factNode_to_actionNode.clear();
    const State initial_state = task_proxy.get_initial_state();

    for (const auto &nptr: nodes) {
        LandmarkNode *node = nptr.get();

        unordered_set<int> op_union;
        op_union.reserve(16);

        const Landmark &lm = node->get_landmark();//fact lm node
        const bool holds_initially = lm.is_true_in_state(initial_state);
        // A parent that doesn't hold in s0 implies this landmark must be re-added later.
        const bool has_parent_not_holding_initially = std::any_of(
            node->parents.begin(), node->parents.end(),
            [&](const std::pair<LandmarkNode *, EdgeType> &par) {
                return !par.first->get_landmark().is_true_in_state(initial_state);
            });

        const bool should_transform =
            !holds_initially || (holds_initially && has_parent_not_holding_initially);

        if (!should_transform)
            continue;
        
        op_union.insert(lm.first_achievers.begin(), lm.first_achievers.end());
        vector<int> opIDs = to_sorted_vector(std::move(op_union));

        //create action lm, action lm node, add node to action lm graph
        ActionLandmark action_lm(vector<int>(opIDs.begin(), opIDs.end()));
        action_lm.possible_achievers = lm.possible_achievers;
        ActionLandmarkNode *new_action_lm_node = &action_lm_graph.add_action_lm(move(action_lm));
        // map fact lm node to action lm node
        factNode_to_actionNode[node] = new_action_lm_node; 
    }
    //set up edges(natural ordering) between action lm nodes
    setUp_edge(); 
    action_lm_graph.set_action_lm_ids(); 
    if(log.is_at_least_normal()) {
        log << "Action landmark graph generation time: " << timer << endl;
        log << "Number of action landmarks: " << action_lm_graph.get_num_action_lms() << endl;
    }
    
}

//For each fact/disj_fact landmark node, union of operator IDs that achieve any atom from that fact landmark
const vector<int> *Transformer::get_action_achievers(const LandmarkNode *node) const {
    auto it = factNode_to_actionNode.find(node);
    if (it == factNode_to_actionNode.end())
        return nullptr;
    return &it->second->get_Actionlandmark().actions;
}


void Transformer::compute_min_costs() {
    const int N = action_lm_graph.get_num_action_lms();
    min_cost.assign(N, 0);

    const int min_operator_cost = task_properties::get_min_operator_cost(task_proxy);

    for (const auto &up : action_lm_graph.get_nodes()) {
        const auto &acts = up->get_Actionlandmark().actions;
        int best = std::numeric_limits<int>::max();

        for (int id : acts) {
            auto it = bank_cost.find(id);
            if (it != bank_cost.end()) {
                best = std::min(best, it->second);
            }
        }
        if (acts.empty() || best == std::numeric_limits<int>::max())
            //best = 0; // empty A contributes 0
            best = min_operator_cost;

        min_cost[up->get_id()] = best;
    }
}

int Transformer::get_min_cost_of_achievers(
    const std::unordered_set<int> &achievers) const {
    int min_cost = numeric_limits<int>::max();
    for (int id : achievers) {
        OperatorProxy op = get_operator_or_axiom(task_proxy, id);
        min_cost = min(min_cost, op.get_cost());
    }
    return min_cost;
}

void Transformer::compute_min_possible_costs() {
    const int N = action_lm_graph.get_num_action_lms();
    possible_min_cost.assign(N, 0);

    for (const auto &up : action_lm_graph.get_nodes()) {
        const auto &possible_achievers = up->get_Actionlandmark().possible_achievers;
        possible_min_cost[up->get_id()] = get_min_cost_of_achievers(possible_achievers);
    }
}

void Transformer::discard_all_orderings() {
    //some landmark factories which extend base landmark factory may support orderings, has a boolean flag 'use_orders' to indicate whether to use orderings,
    //but base landmark factory does not have this flag

    if(lm_factory->orders_enabled()) 
        return;

    if (log.is_at_least_normal()) {
            log << "Discarding all orderings in ActionLM" << endl;
    }
    for (auto &node_ptr : action_lm_graph.get_nodes()) {
        ActionLandmarkNode *node = node_ptr.get();
        node->parents.clear();
        node->children.clear();
    }
} 

void Transformer::edge_add(ActionLandmarkNode &from, ActionLandmarkNode &to, EdgeType type) {
    // Accept only NATURAL-or-weaker orderings
    if (type > EdgeType::NATURAL) {
        if (log.is_at_least_normal()) {
            log << "Only NATURAL or weaker orderings are supported in ActionLM" << endl;
        }
        return;
    }

    // If edge already exists, remove if weaker
    if (from.children.find(&to) != from.children.end() && from.children.find(
            &to)->second < type) {
        from.children.erase(&to);
        assert(to.parents.find(&from) != to.parents.end());
        to.parents.erase(&from);

        assert(to.parents.find(&from) == to.parents.end());
        assert(from.children.find(&to) == from.children.end());
    }
    // If edge does not exist (or has just been removed), insert
    if (from.children.find(&to) == from.children.end()) {
        assert(to.parents.find(&from) == to.parents.end());
        from.children.emplace(&to, type);
        to.parents.emplace(&from, type);
    }
}

void Transformer::setUp_edge() {
    bool any_parents = false;
    for (const auto &fact_node_ptr : lm_graph->get_nodes()) {
        if (!fact_node_ptr->parents.empty()) { any_parents = true; break; }
    }
    if (!any_parents) {
        if (log.is_at_least_normal()) log << "No orderings in fact LM; No Action LM edges will be enabled \n";
        return; // nothing to add
    }
    //nodes: vector<std::unique_ptr<LandmarkNode>>    
    for(const auto &fact_node_ptr: lm_graph->get_nodes()) {
        //get the corresponding action node of this fact lm node
        LandmarkNode *fact_node = fact_node_ptr.get();
        auto it_to = factNode_to_actionNode.find(fact_node);
        if(it_to == factNode_to_actionNode.end())
            continue;
        ActionLandmarkNode *to = it_to->second;

        for(auto &pred : fact_node->parents) { // fact_node->parents is unordered_map<LandmarkNode*, EdgeType> which are (predecessors) of this fact lm node
            LandmarkNode *fact_pred = pred.first;
            auto it_from = factNode_to_actionNode.find(fact_pred);
            if (it_from == factNode_to_actionNode.end())
                continue;
            ActionLandmarkNode *from = it_from->second;
            edge_add(*from, *to, EdgeType::NATURAL); 
        }
    }
}

} 
