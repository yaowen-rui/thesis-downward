//
// Created by rui on 15.08.2025.
//
#ifndef LANDMARKS_LANDMARK_FACTORY_DISJUNCTIVE_ACTIONLM_H
#define LANDMARKS_LANDMARK_FACTORY_DISJUNCTIVE_ACTIONLM_H

#include "landmark_factory_relaxation.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace landmarks {
class LandmarkFactoryDisjunctiveActionLM : public LandmarkFactoryRelaxation {
  const bool disjunctive_landmarks;
  const bool use_orders;
  const bool only_causal_landmarks;
  std::list<LandmarkNode *> open_landmarks;
  std::vector<std::vector<int>> disjunction_classes;

  std::unordered_map<LandmarkNode *, utils::HashSet<FactPair>> forward_orders;

  // dtg_successors[var_id][val] contains all successor values of val in the
  // domain transition graph for the variable
  std::vector<std::vector<std::unordered_set<int>>> dtg_successors;

  /*Disjunctive action landmark: For each disjunctive fact landmark node, we store the union of operator IDs that
   * can achieve any member atom of that disjunction */
  std::unordered_map<const LandmarkNode *, std::vector<int>> disj_action_achievers;
  //single index for disj action landmarks keyed by e.g. "1,7,12" signature
  std::unordered_map<std::string, LandmarkNode *> action_nodes_by_sig;

  void build_dtg_successors(const TaskProxy &task_proxy);
  void add_dtg_successor(int var_id, int pre, int post);
  void find_forward_orders(const VariablesProxy &variables,
                           const std::vector<std::vector<bool>> &reached,
                           LandmarkNode *lm_node);
  void add_lm_forward_orders();

  void get_greedy_preconditions_for_lm(
      const TaskProxy &task_proxy, const Landmark &landmark,
      const OperatorProxy &op,
      std::unordered_map<int, int> &result) const;

  //if all achievers of a landmark share the same single fact as a precondition, 
  //that fact is promoted as a simple landmark.   
  void compute_shared_preconditions(
      const TaskProxy &task_proxy,
      std::unordered_map<int, int> &shared_pre,
      std::vector<std::vector<bool>> &reached, const Landmark &landmark);
  //if achievers don’t share the exact same fact, but each achiever 
  //contributes one fact from the same disjunction class (same predicate), 
  //then the union of those facts is created as a disjunctive landmark.
  void compute_disjunctive_preconditions(
      const TaskProxy &task_proxy,
      std::vector<std::set<FactPair>> &disjunctive_pre,
      std::vector<std::vector<bool>> &reached,
      const Landmark &landmark);

  virtual void generate_relaxed_landmarks(
      const std::shared_ptr<AbstractTask> &task,
      Exploration &exploration) override;
  void found_simple_lm_and_order(const FactPair &a, LandmarkNode &b,
                                 EdgeType t);
  void found_disj_lm_and_order(const TaskProxy &task_proxy,
                               const std::set<FactPair> &a,
                               LandmarkNode &b,
                               EdgeType t);
  void approximate_lookahead_orders(const TaskProxy &task_proxy,
                                    const std::vector<std::vector<bool>> &reached,
                                    LandmarkNode *lmp);
  bool domain_connectivity(const State &initial_state,
                           const FactPair &landmark,
                           const std::unordered_set<int> &exclude);

  void build_disjunction_classes(const TaskProxy &task_proxy);

  void discard_disjunctive_landmarks();



public:
  LandmarkFactoryDisjunctiveActionLM(
      bool disjunctive_landmarks, bool use_orders,
      bool only_causal_landmarks, utils::Verbosity verbosity);

  virtual bool supports_conditional_effects() const override;

  //return nullptr if the nodes has no stored action achievers;
  const std::vector<int> *get_disj_action_achievers(
      const LandmarkNode *node) const {
      auto it = disj_action_achievers.find(node);
      return (it==disj_action_achievers.end())? nullptr : &it->second;
  };

private:
  //compute and attach L^a = union of achievers for all atoms in a disjunctive fact landmark
  void attach_disj_action_achievers (LandmarkNode *lm_node,
                                    const std::set<FactPair> &atoms);

  //convert an unordered set of operator IDs into a sorted
  static std::vector<int> to_sorted_vector(std::unordered_set<int> &&s);

  /* build a stable key like "3,9,12" for indexing operatior sets: 
  * action_union_signature(ops) takes the union of achiever operator IDs for a single disjunctive fact LM node, 
  *sorts and de-duplicate them, then joins them with commas to make one stable string key.
  Same op-set -> same signature -> reuse the same action-LM node across multiple fact LMs.
  Different op-sets -> different signatures -> distinct action-LM nodes.
  */
  static std::string action_union_signature(const std::vector<int> &ops);
};
}

#endif

