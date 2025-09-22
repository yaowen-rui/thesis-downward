#ifndef LANDMARK_SUM_HEURISTIC_DAL_H
#define LANDMARK_SUM_HEURISTIC_DAL_H

#include "landmark_heuristic.h"
#include "landmark_status_manager_action.h"
#include "transformer.h"

namespace landmarks {
class LandmarkSumHeuristicDal : public LandmarkHeuristic {
private:
  Transformer transformer;

  LandmarkStatusManagerAction lm_status_manager_action;

  const bool dead_ends_reliable=true;

  // Cached per-action-landmark minimal operator cost.
  // Indexed by action-LM node id.
  std::vector<int> min_costs_per_action_lm;

  void initialize_costs();
  void setup_transformer(const std::shared_ptr<LandmarkFactory> &lm_factory);

protected:
  int get_heuristic_value(const State &ancestor_state) override;
  
public:
  LandmarkSumHeuristicDal(
      const std::shared_ptr<LandmarkFactory> &lm_factory,//use lm_factory to build transformer, them create action lm graph
      bool /*pref*/, bool /*prog_goal*/, bool /*prog_gn*/, bool /*prog_r*/,
      const std::shared_ptr<AbstractTask> &task_transform,
      bool cache_estimates, const std::string &description,
      utils::Verbosity verbosity, tasks::AxiomHandlingType axioms);

  bool dead_ends_are_reliable() const override {return dead_ends_reliable;};

  int compute_heuristic(const State &ancestor_state) override;
  void notify_initial_state(const State &initial_state) override;
  void notify_state_transition(const State &parent_state,
                             OperatorID op_id,
                             const State &state) override;
};
}

#endif 
