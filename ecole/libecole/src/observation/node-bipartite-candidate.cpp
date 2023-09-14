#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <scip/scip.h>
#include <scip/struct_lp.h>
#include <xtensor/xview.hpp>
#include <range/v3/view/zip.hpp>

#include "ecole/observation/node-bipartite-candidate.hpp"
#include "ecole/scip/model.hpp"
#include "ecole/scip/row.hpp"
#include "ecole/utility/unreachable.hpp"

namespace ecole::observation {

namespace {

namespace views = ranges::views;

/*********************
 *  Common helpers   *
 *********************/

using xmatrix = decltype(NodeBipartiteCandObs::column_features);
using value_type = xmatrix::value_type;

using ColumnFeatures = NodeBipartiteCandObs::ColumnFeatures;

value_type constexpr cste = 5.;
value_type constexpr nan = std::numeric_limits<value_type>::quiet_NaN();

double root_reduced_obj_norm = -1;

auto row_get_reduced_norm = [](auto const row) {
	double row_norm = 0;
	auto const row_cols = scip::get_cols(row);
	auto const row_values = scip::get_vals(row);
		
	for (auto const [val, col] : views::zip(row_values, row_cols))  { 
		if(SCIPcolGetLPPos(col) >= 0) {
			if (SCIPcolGetLb(col) + 0.5 <= SCIPcolGetUb(col)) //not fixed column!
				row_norm += val * val;
		}
	}
	return sqrt(row_norm);
};

SCIP_Real obj_l2_norm(SCIP* const scip) noexcept {
	auto const norm = SCIPgetObjNorm(scip);
	return norm > 0 ? norm : 1.;
}

SCIP_Real reduced_obj_l2_norm(scip::Model& model) noexcept {

	double reduced_obj_norm = 0.0;
	auto const lp_columns = model.lp_columns();
	for (auto* const col : lp_columns) {

		if (SCIPcolGetLb(col) + 0.5 <= SCIPcolGetUb(col))
			reduced_obj_norm += (SCIPcolGetObj(col) * SCIPcolGetObj(col));

	}
	reduced_obj_norm = sqrt(reduced_obj_norm);
	return reduced_obj_norm > 0 ? reduced_obj_norm : 1.;
}




SCIP_Real row_l2_norm(SCIP_ROW* const row) noexcept {
	auto const norm = SCIProwGetNorm(row);
	return norm > 0 ? norm : 1.;
}

SCIP_Real reduced_row_l2_norm(SCIP_ROW* const row) noexcept {
	auto const norm = row_get_reduced_norm(row);
	return norm > 0 ? norm : 1.;
}

SCIP_Real obj_cos_sim(SCIP* const scip, SCIP_ROW* const row) noexcept {
	auto const norm_prod = SCIProwGetNorm(row) * SCIPgetObjNorm(scip);
	if (SCIPisPositive(scip, norm_prod)) {
		return row->objprod / norm_prod;
	}
	return 0.;
}

SCIP_Real reduced_obj_cos_sim(SCIP* const scip, SCIP_ROW* const row, double reduced_obj_norm) noexcept {

	auto const norm_prod = row_get_reduced_norm(row) * reduced_obj_norm;

	double inner_product = 0.0;

	auto const row_values = scip::get_vals(row);

	auto const row_cols = scip::get_cols(row);

	for (auto const [val, col] : views::zip(row_values, row_cols))  { 
		if(SCIPcolGetLPPos(col) >= 0) {
			if (SCIPcolGetLb(col) + 0.5 <= SCIPcolGetUb(col)) //not fixed column!
				inner_product += val * SCIPcolGetObj(col);
		}
	}
	
	if (SCIPisPositive(scip, norm_prod)) {
		return inner_product / norm_prod;
	}
	return 0.;
}

/******************************************
 *  Column features extraction functions  *
 ******************************************/
std::optional<SCIP_Real> upper_bound(SCIP* const scip, SCIP_COL* const col) noexcept {
	auto const ub_val = SCIPcolGetUb(col);
	if (SCIPisInfinity(scip, std::abs(ub_val))) {
		return {};
	}
	return ub_val;
}

std::optional<SCIP_Real> lower_bound(SCIP* const scip, SCIP_COL* const col) noexcept {
	auto const lb_val = SCIPcolGetLb(col);
	if (SCIPisInfinity(scip, std::abs(lb_val))) {
		return {};
	}
	return lb_val;
}

bool is_prim_sol_at_lb(SCIP* const scip, SCIP_COL* const col) noexcept {
	auto const lb_val = lower_bound(scip, col);
	if (lb_val) {
		return SCIPisEQ(scip, SCIPcolGetPrimsol(col), lb_val.value());
	}
	return false;
}

bool is_prim_sol_at_ub(SCIP* const scip, SCIP_COL* const col) noexcept {
	auto const ub_val = upper_bound(scip, col);
	if (ub_val) {
		return SCIPisEQ(scip, SCIPcolGetPrimsol(col), ub_val.value());
	}
	return false;
}

std::optional<SCIP_Real> best_sol_val(SCIP* const scip, SCIP_VAR* const var) noexcept {
	auto* const sol = SCIPgetBestSol(scip);
	if (sol != nullptr) {
		return SCIPgetSolVal(scip, sol, var);
	}
	return {};
}

std::optional<SCIP_Real> avg_sol(SCIP* const scip, SCIP_VAR* const var) noexcept {
	if (SCIPgetBestSol(scip) != nullptr) {
		return SCIPvarGetAvgSol(var);
	}
	return {};
}

std::optional<SCIP_Real> feas_frac(SCIP* const scip, SCIP_VAR* const var) noexcept {
	if (SCIPvarGetType(var) == SCIP_VARTYPE_CONTINUOUS) {
		return {};
	}
	return SCIPfeasFrac(scip, SCIPvarGetLPSol(var));
}

std::optional<SCIP_Real> feas_infeasibility(SCIP* const scip, SCIP_VAR* const var) noexcept {
	if (SCIPvarGetType(var) == SCIP_VARTYPE_CONTINUOUS) {
		return {};
	}

	return MIN(SCIPvarGetLPSol(var) - SCIPfeasFloor(scip, SCIPvarGetLPSol(var)), SCIPfeasCeil(scip, SCIPvarGetLPSol(var)) - SCIPvarGetLPSol(var));

}

/** Convert an enum to its underlying index. */
template <typename E> constexpr auto idx(E e) {
	return static_cast<std::underlying_type_t<E>>(e);
}

template <typename Features>
void set_static_features_for_col(Features&& out, SCIP_VAR* const var, SCIP_COL* const col, value_type obj_norm) {
	out[idx(ColumnFeatures::objective)] = SCIPcolGetObj(col) / obj_norm; //reduced norm
	// On-hot enconding of variable type
	out[idx(ColumnFeatures::is_type_binary)] = 0.;
	out[idx(ColumnFeatures::is_type_integer)] = 0.;
	out[idx(ColumnFeatures::is_type_implicit_integer)] = 0.;
	out[idx(ColumnFeatures::is_type_continuous)] = 0.;
	switch (SCIPvarGetType(var)) {
	case SCIP_VARTYPE_BINARY:
		out[idx(ColumnFeatures::is_type_binary)] = 1.;
		break;
	case SCIP_VARTYPE_INTEGER:
		out[idx(ColumnFeatures::is_type_integer)] = 1.;
		break;
	case SCIP_VARTYPE_IMPLINT:
		out[idx(ColumnFeatures::is_type_implicit_integer)] = 1.;
		break;
	case SCIP_VARTYPE_CONTINUOUS:
		out[idx(ColumnFeatures::is_type_continuous)] = 1.;
		break;
	default:
		utility::unreachable();
	}

}

template <typename Features>
void set_dynamic_features_for_col(
	Features&& out,
	SCIP* const scip,
	SCIP_VAR* const var,
	SCIP_COL* const col,
	value_type reduced_obj_norm,
	value_type n_lps) {

	out[idx(ColumnFeatures::has_lower_bound)] = static_cast<value_type>(lower_bound(scip, col).has_value());
	out[idx(ColumnFeatures::has_upper_bound)] = static_cast<value_type>(upper_bound(scip, col).has_value());
	out[idx(ColumnFeatures::normed_reduced_cost)] = SCIPgetVarRedcost(scip, var) / reduced_obj_norm;
	out[idx(ColumnFeatures::solution_value)] = SCIPvarGetLPSol(var);
	out[idx(ColumnFeatures::solution_frac)] = feas_frac(scip, var).value_or(0.);
	out[idx(ColumnFeatures::is_solution_at_lower_bound)] = static_cast<value_type>(is_prim_sol_at_lb(scip, col));
	out[idx(ColumnFeatures::is_solution_at_upper_bound)] = static_cast<value_type>(is_prim_sol_at_ub(scip, col));
	out[idx(ColumnFeatures::scaled_age)] = static_cast<value_type>(SCIPcolGetAge(col)) / (n_lps + cste);
	out[idx(ColumnFeatures::incumbent_value)] = best_sol_val(scip, var).value_or(nan);
	out[idx(ColumnFeatures::average_incumbent_value)] = avg_sol(scip, var).value_or(nan);
	// On-hot encoding
	out[idx(ColumnFeatures::is_basis_lower)] = 0.;
	out[idx(ColumnFeatures::is_basis_basic)] = 0.;
	out[idx(ColumnFeatures::is_basis_upper)] = 0.;
	out[idx(ColumnFeatures::is_basis_zero)] = 0.;
	switch (SCIPcolGetBasisStatus(col)) {
	case SCIP_BASESTAT_LOWER:
		out[idx(ColumnFeatures::is_basis_lower)] = 1.;
		break;
	case SCIP_BASESTAT_BASIC:
		out[idx(ColumnFeatures::is_basis_basic)] = 1.;
		break;
	case SCIP_BASESTAT_UPPER:
		out[idx(ColumnFeatures::is_basis_upper)] = 1.;
		break;
	case SCIP_BASESTAT_ZERO:
		out[idx(ColumnFeatures::is_basis_zero)] = 1.;
		break;
	default:
		utility::unreachable();
	}

	// New features: infeasibility of solution: ex: For 2.8, min(2.8 - 2, 3 - 2.8) = 0.2
	out[idx(ColumnFeatures::solution_infeasibility)] = feas_infeasibility(scip, var).value_or(0.);

	// One-hot encoding

	SCIP_Real* myvals = SCIPcolGetVals(col);
	SCIP_ROW** myrows = SCIPcolGetRows(col);
	SCIP_ROW* row = nullptr;
	int nlprows = SCIPcolGetNLPNonz(col);

	SCIP_Real edge_sum = 0;
	SCIP_Real edge_min = SCIPinfinity(scip);
	SCIP_Real edge_max = -SCIPinfinity(scip);
	SCIP_Real edge_val;

	SCIP_Real bias_sum = 0;
	SCIP_Real bias_min = SCIPinfinity(scip);
	SCIP_Real bias_max = -SCIPinfinity(scip);
	SCIP_Real bias_val;

	SCIP_Real obj_cos_sim_sum = 0;
	SCIP_Real obj_cos_sim_min = 1.0;
	SCIP_Real obj_cos_sim_max = -1.0;
	SCIP_Real obj_cos_sim_val;

	SCIP_Real is_tight_sum = 0;
	SCIP_Real is_tight_min = 1.0;
	SCIP_Real is_tight_max = -1.0;
	SCIP_Real is_tight_val;

	SCIP_Real dual_solution_sum = 0;
	SCIP_Real dual_solution_min = SCIPinfinity(scip);
	SCIP_Real dual_solution_max = -SCIPinfinity(scip);
	SCIP_Real dual_solution_val;

	SCIP_Real scaled_age_sum = 0;
	SCIP_Real scaled_age_min = SCIPinfinity(scip);
	SCIP_Real scaled_age_max = -SCIPinfinity(scip);
	SCIP_Real scaled_age_val;

	int row_counter = 0;
	SCIP_Real row_norm;

	for (int row_idx = 0; row_idx < nlprows; ++row_idx)
	{
		row = myrows[row_idx];
		row_norm = reduced_row_l2_norm(row);

		auto const row_cols = scip::get_cols(row);
		auto const row_values = scip::get_vals(row);
		double temp_sum = 0;
		for (auto const [val, row_col] : views::zip(row_values, row_cols))  { 
			if(SCIPcolGetLPPos(row_col) >= 0) {
				if (SCIPcolGetLb(row_col) + 0.5 > SCIPcolGetUb(row_col)) //fixed column!
					temp_sum += val * SCIPcolGetLb(row_col); //value..
			}
		}
		
		if ( scip::get_unshifted_lhs(scip, row).has_value() )  {
			
			edge_val = -myvals[row_idx] / row_norm; // in [-1,1]
			edge_sum += edge_val;
			edge_min = MIN(edge_min, edge_val);
			edge_max = MAX(edge_max, edge_val);

			bias_val = -1. * (scip::get_unshifted_lhs(scip, row).value() - temp_sum) / row_norm;
			bias_sum += bias_val;
			bias_min = MIN(bias_min, bias_val);
			bias_max = MAX(bias_max, bias_val);

			obj_cos_sim_val = -1 * reduced_obj_cos_sim(scip,row, reduced_obj_norm);
			obj_cos_sim_sum += obj_cos_sim_val;
			obj_cos_sim_min = MIN(obj_cos_sim_min, obj_cos_sim_val);
			obj_cos_sim_max = MAX(obj_cos_sim_max, obj_cos_sim_val);

			is_tight_val = static_cast<value_type>(scip::is_at_lhs(scip, row));
			is_tight_sum += is_tight_val;
			is_tight_min = MIN(is_tight_min, is_tight_val);
			is_tight_max = MAX(is_tight_max, is_tight_val);

			dual_solution_val = -1. * SCIProwGetDualsol(row) / (row_norm * reduced_obj_norm);
			dual_solution_sum += dual_solution_val;
			dual_solution_min = MIN(dual_solution_min, dual_solution_val);
			dual_solution_max = MAX(dual_solution_max, dual_solution_val);

			scaled_age_val = static_cast<value_type>(SCIProwGetAge(row)) / (n_lps + cste);
			scaled_age_sum += scaled_age_val;
			scaled_age_min = MIN(scaled_age_min, scaled_age_val);
			scaled_age_max = MAX(scaled_age_max, scaled_age_val);

			row_counter ++;

		}
		if ( scip::get_unshifted_rhs(scip, row).has_value() )  {
			
			edge_val = myvals[row_idx] / row_norm; // in [-1,1]
			edge_sum += edge_val;
			edge_min = MIN(edge_min, edge_val);
			edge_max = MAX(edge_max, edge_val);

			bias_val = (scip::get_unshifted_rhs(scip, row).value() - temp_sum) / row_norm;
			bias_sum += bias_val;
			bias_min = MIN(bias_min, bias_val);
			bias_max = MAX(bias_max, bias_val);

			obj_cos_sim_val = reduced_obj_cos_sim(scip,row, reduced_obj_norm);
			obj_cos_sim_sum += obj_cos_sim_val;
			obj_cos_sim_min = MIN(obj_cos_sim_min, obj_cos_sim_val);
			obj_cos_sim_max = MAX(obj_cos_sim_max, obj_cos_sim_val);

			is_tight_val = static_cast<value_type>(scip::is_at_rhs(scip, row));
			is_tight_sum += is_tight_val;
			is_tight_min = MIN(is_tight_min, is_tight_val);
			is_tight_max = MAX(is_tight_max, is_tight_val);

			dual_solution_val = SCIProwGetDualsol(row) / (row_norm * reduced_obj_norm);
			dual_solution_sum += dual_solution_val;
			dual_solution_min = MIN(dual_solution_min, dual_solution_val);
			dual_solution_max = MAX(dual_solution_max, dual_solution_val);

			scaled_age_val = static_cast<value_type>(SCIProwGetAge(row)) / (n_lps + cste);
			scaled_age_sum += scaled_age_val;
			scaled_age_min = MIN(scaled_age_min, scaled_age_val);
			scaled_age_max = MAX(scaled_age_max, scaled_age_val);
			
			row_counter ++;

		}
	}
	/*
	if (out[idx(ColumnFeatures::solution_frac)] > 0)
		printf("Edge mean %.2f min %.2f max %.2f \n", edge_sum / row_counter, edge_min, edge_max);
	*/
	out[idx(ColumnFeatures::edge_mean)] = edge_sum / row_counter;
	out[idx(ColumnFeatures::edge_min)] = edge_min;
	out[idx(ColumnFeatures::edge_max)] = edge_max;

	out[idx(ColumnFeatures::bias_mean)] = bias_sum / row_counter;
	out[idx(ColumnFeatures::bias_min)] = bias_min;
	out[idx(ColumnFeatures::bias_max)] = bias_max;

	out[idx(ColumnFeatures::obj_cos_sim_mean)] = obj_cos_sim_sum / row_counter;
	out[idx(ColumnFeatures::obj_cos_sim_min)] = obj_cos_sim_min;
	out[idx(ColumnFeatures::obj_cos_sim_max)] = obj_cos_sim_max;

	out[idx(ColumnFeatures::is_tight_mean)] = is_tight_sum / row_counter;
	out[idx(ColumnFeatures::is_tight_min)] = is_tight_min;
	out[idx(ColumnFeatures::is_tight_max)] = is_tight_max;

	out[idx(ColumnFeatures::dual_solution_mean)] = dual_solution_sum / row_counter;
	out[idx(ColumnFeatures::dual_solution_min)] = dual_solution_min;
	out[idx(ColumnFeatures::dual_solution_max)] = dual_solution_max;

	out[idx(ColumnFeatures::scaled_age_mean)] = scaled_age_sum / row_counter;
	out[idx(ColumnFeatures::scaled_age_min)] = scaled_age_min;
	out[idx(ColumnFeatures::scaled_age_max)] = scaled_age_max;

}

void set_features_for_all_cols(xmatrix& out, scip::Model& model, bool const update_static) {
	auto* const scip = model.get_scip_ptr();
	if(SCIPgetCurrentNode(scip) == SCIPgetRootNode(scip))
	{
		root_reduced_obj_norm = reduced_obj_l2_norm(model);
	}
	// Contant reused in every iterations
	auto const n_lps = static_cast<value_type>(SCIPgetNLPs(scip));
	auto reduced_obj_norm = reduced_obj_l2_norm(model);
	auto const columns = model.lp_columns();
	auto const n_columns = columns.size();
	for (std::size_t col_idx = 0; col_idx < n_columns; ++col_idx) {
		auto* const col = columns[col_idx];
		auto* const var = SCIPcolGetVar(col);
		auto features = xt::row(out, static_cast<std::ptrdiff_t>(col_idx));
		if (update_static) {
			set_static_features_for_col(features, var, col, root_reduced_obj_norm);
		}
		set_dynamic_features_for_col(features, scip, var, col, reduced_obj_norm, n_lps);
	}
}

auto is_on_root_node(scip::Model& model) -> bool {
	auto* const scip = model.get_scip_ptr();
	return SCIPgetCurrentNode(scip) == SCIPgetRootNode(scip);
}

auto extract_observation_fully(scip::Model& model) -> NodeBipartiteCandObs {
	auto obs = NodeBipartiteCandObs{
		xmatrix::from_shape({model.lp_columns().size(), NodeBipartiteCandObs::n_column_features}),
	};
	set_features_for_all_cols(obs.column_features, model, true);
	return obs;
}

auto extract_observation_from_cache(scip::Model& model, NodeBipartiteCandObs obs) -> NodeBipartiteCandObs {
	set_features_for_all_cols(obs.column_features, model, false);
	return obs;
}

}  // namespace

/*************************************
 *  Observation extracting function  *
 *************************************/

auto NodeBipartiteCand::before_reset(scip::Model& /* model */) -> void {
	cache_computed = false;
}

auto NodeBipartiteCand::extract(scip::Model& model, bool /* done */) -> std::optional<NodeBipartiteCandObs> {
	if (model.stage() == SCIP_STAGE_SOLVING) {
		if (use_cache) {
			if (is_on_root_node(model)) {
				the_cache = extract_observation_fully(model);
				cache_computed = true;
				return the_cache;
			}
			if (cache_computed) {
				return extract_observation_from_cache(model, the_cache);
			}
		}
		return extract_observation_fully(model);
	}
	return {};
}

}  // namespace ecole::observation
