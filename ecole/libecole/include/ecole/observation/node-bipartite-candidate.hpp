#pragma once

#include <optional>

#include <xtensor/xtensor.hpp>

#include "ecole/export.hpp"
#include "ecole/observation/abstract.hpp"
#include "ecole/utility/sparse-matrix.hpp"

namespace ecole::observation {

struct ECOLE_EXPORT NodeBipartiteCandObs {
	using value_type = double;

	static inline std::size_t constexpr n_static_column_features = 5; // 5
	static inline std::size_t constexpr n_dynamic_column_features = 33; //14
	static inline std::size_t constexpr n_column_features = n_static_column_features + n_dynamic_column_features;
	enum struct ECOLE_EXPORT ColumnFeatures : std::size_t {
		/** Static features */
		objective = 0,
		is_type_binary,            // One hot encoded
		is_type_integer,           // One hot encoded
		is_type_implicit_integer,  // One hot encoded
		is_type_continuous,        // One hot encoded


		/** Dynamic features */
		has_lower_bound,
		has_upper_bound,
		normed_reduced_cost,
		solution_value,
		solution_frac,
		is_solution_at_lower_bound,
		is_solution_at_upper_bound,
		scaled_age,
		incumbent_value,
		average_incumbent_value,
		is_basis_lower,
		is_basis_basic,
		is_basis_upper,
		is_basis_zero,
		solution_infeasibility, // new feature
		edge_mean,
		edge_min,
		edge_max,
		bias_mean,
		bias_min,
		bias_max,
		obj_cos_sim_mean,
		obj_cos_sim_min,
		obj_cos_sim_max,
		is_tight_mean,
		is_tight_min,
		is_tight_max,
		dual_solution_mean,
		dual_solution_min,
		dual_solution_max,
		scaled_age_mean,
		scaled_age_min,
		scaled_age_max,

	};


	xt::xtensor<value_type, 2> column_features;

};

class ECOLE_EXPORT NodeBipartiteCand {
public:
	NodeBipartiteCand(bool cache = false) : use_cache{cache} {}

	ECOLE_EXPORT auto before_reset(scip::Model& model) -> void;

	ECOLE_EXPORT auto extract(scip::Model& model, bool done) -> std::optional<NodeBipartiteCandObs>;

private:
	NodeBipartiteCandObs the_cache;
	bool use_cache = false;
	bool cache_computed = false;
};

}  // namespace ecole::observation
