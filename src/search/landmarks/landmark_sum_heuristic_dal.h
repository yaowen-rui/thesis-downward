//
// Created by rui on 15.08.2025.
//

#ifndef FAST_DOWNWARD_LANDMARK_SUM_HEURISTIC_DAL_H
#define FAST_DOWNWARD_LANDMARK_SUM_HEURISTIC_DAL_H

#include "landmark_heuristic.h"

namespace landmarks {
class LandmarkFactoryDisjunctiveActionLM; //forward declaration
}

namespace landmarks {
class LandmarkSumHeuristicDal : public LandmarkHeuristic {
  const bool dead_ends_reliable;

  std::vector<int> min_first_achiever_costs;
  std::vector<int> min_possible_achiever_costs;

  //for action layer
  std::vector<int> min_action_costs;// precompute min operator cost for each ACTION node id
  std::vector<unsigned char> is_action_node;//for each node id: 1 if it's an action lm, else 0
  bool use_action_layer = false;//true iff graph contains any action lm nodes
  int get_min_cost_of_achievers(const std::vector<int> &op_ids);

  int get_min_cost_of_achievers(
      const std::unordered_set<int> &achievers) const;

  void compute_landmark_costs();

  int get_heuristic_value(const State &ancestor_state) override;
public:
  LandmarkSumHeuristicDal(
      const std::shared_ptr<LandmarkFactory> &lm_factory, bool pref,
      bool prog_goal, bool prog_gn, bool prog_r,
      const std::shared_ptr<AbstractTask> &transform,
      bool cache_estimates, const std::string &description,
      utils::Verbosity verbosity, tasks::AxiomHandlingType axioms);

  virtual bool dead_ends_are_reliable() const override;
};
}

#endif 
