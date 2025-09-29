
#include "landmark_sum_heuristic_dal.h"

#include "landmark.h"

#include "util.h"

#include "../plugins/plugin.h"
#include "../task_utils/task_properties.h"
#include "../utils/markup.h"
#include "../abstract_task.h"

#include <limits>

using namespace std;

namespace landmarks {

LandmarkSumHeuristicDal::LandmarkSumHeuristicDal(
  const shared_ptr<LandmarkFactory> &lm_factory,
  bool /*pref*/, bool /*prog_goal*/, bool /*prog_gn*/, bool /*prog_r*/,
  const shared_ptr<AbstractTask> &task_transform,
  bool cache_estimates, const string &description,
  utils::Verbosity verbosity, tasks::AxiomHandlingType axioms)
  : LandmarkHeuristic(
    /*use_preferred_operators=*/false,
    tasks::get_default_value_axioms_task_if_needed(task_transform, axioms),
    cache_estimates, description, verbosity),
    //build action lm structure 
    transformer(task_proxy, lm_factory, task_transform),
    lm_status_manager_action(transformer) {
      if (log.is_at_least_normal()) {
          log << "Initializing Action Landmark Sum Heuristic..." << endl;
      }

      initialize_costs();
      
      // Initialize action-landmark past/future for the initial state:
      // past = ∅, future = All.
      //State init = task_proxy.get_initial_state();
      //lm_status_manager_action.progress_initial_state(init);
  }

void LandmarkSumHeuristicDal::initialize_costs() {
  min_costs_per_action_lm = transformer.get_min_cost();

}

int LandmarkSumHeuristicDal::compute_edge_local_h(const State &parent_state, OperatorID op_id) const {
    // Start from parent's future bitset
    ConstBitsetView future0 = lm_status_manager_action.get_future_action_landmarks(parent_state);
    const auto &op2LMs = lm_status_manager_action.get_op_to_actionLMs();
    int op = op_id.get_index();

    // Build edge-local future := future(parent) \ Hit(op)
    const int n = static_cast<int>(transformer.action_lm_graph.get_num_action_lms());
    std::vector<char> future_edge(n, 0);
    for (int i = 0; i < n; ++i)
        future_edge[i] = future0.test(i) ? 1 : 0;

    if (op >= 0 && op < static_cast<int>(op2LMs.size())) {
        for (size_t A : op2LMs[op]) {
            if (A >= 0 && A < (size_t)n)
                future_edge[static_cast<int>(A)] = 0;
        }
    }

    // Sum costs over remaining future_edge
    int h_edge = 0;
    for (int i = 0; i < n; ++i) {
        if (future_edge[i])
            h_edge += min_costs_per_action_lm[i];
    }
    return h_edge;
}

int LandmarkSumHeuristicDal::get_heuristic_value(const State &ancestor_state) {
    int h = 0;
    ConstBitsetView futureA =
        lm_status_manager_action.get_future_action_landmarks(ancestor_state);
    const int n = static_cast<int>(transformer.action_lm_graph.get_num_action_lms());
    for (int id = 0; id < n; ++id) {
        if (!futureA.test(id))
            continue;

        const int c = min_costs_per_action_lm[id];
        h += c;
    }
    return h;
   
}

int LandmarkSumHeuristicDal::compute_heuristic(const State &ancestor_state) {
  const int sid = ancestor_state.get_id().get_value(); 
  auto it = best_h_by_state.find(sid);
  if (it != best_h_by_state.end())
      return it->second;
  const int h = get_heuristic_value(ancestor_state);
  return h;
}

void LandmarkSumHeuristicDal::notify_initial_state(const State &initial_state) {
    lm_status_manager_action.progress_initial_state(initial_state);
    // Seed best h for the initial state
    const int sid0 = initial_state.get_id().get_value(); 
    best_h_by_state[sid0] = get_heuristic_value(initial_state);
}

void LandmarkSumHeuristicDal::notify_state_transition(
        const State &parent_state, OperatorID op_id, const State &state) {
    //1. Update status (this will prune violating edges internally)
    lm_status_manager_action.progress(parent_state, op_id, state);
    //2. Compute edge-local scalar h and min-merge it into child's cache
    const int h_edge = compute_edge_local_h(parent_state, op_id);
    const int sid = state.get_id().get_value(); 
    auto it = best_h_by_state.find(sid);
    if (it == best_h_by_state.end()) {
        best_h_by_state.emplace(sid, h_edge);// first parent
    } else {
        it->second = std::min(it->second, h_edge);// better parent
    }
}

class LandmarkSumHeuristicDalFeature : public plugins::TypedFeature<Evaluator, LandmarkSumHeuristicDal> {
public:
  LandmarkSumHeuristicDalFeature() : TypedFeature("lm_sum_action") {
    
    document_title("Landmark sum heuristic (Action-Landmark variant)");
    document_synopsis(
      "h^sum over action landmarks. Builds an Action-LM graph via a transformer "
      "and maintains past/future only for action landmarks during search.");

    // Reuse the standard landmark-heuristic options block 
    add_landmark_heuristic_options_to_feature(*this, "landmark_sum_action_heuristic");
    // Axioms handling option (same as classic)
    tasks::add_axioms_option_to_feature(*this);

    document_language_support("action costs", "supported");
    document_language_support("conditional_effects",
      "supported if the LandmarkFactory supports them; otherwise ignored");
    document_language_support("axioms", "supported");

    document_property("admissible", "no");
    document_property("consistent", "no");
    document_property("safe", "yes (same caveats as classic)");
  }

  std::shared_ptr<LandmarkSumHeuristicDal> create_component(const plugins::Options &opts,
                   const utils::Context &) const override {
    
    return plugins::make_shared_from_arg_tuples<LandmarkSumHeuristicDal>(
      get_landmark_heuristic_arguments_from_options(opts),
      tasks::get_axioms_arguments_from_options(opts));
  }
};

static plugins::FeaturePlugin<LandmarkSumHeuristicDalFeature> _plugin;

}

//use the command to test: ./fast-downward.py misc/tests/benchmarks/miconic/s1-0.pddl --search "astar(landmark_sum_action_heuristic(lm_rhw()))"