#include <cmath>

#include <catch2/catch.hpp>
#include <scip/scip.h>

#include "ecole/observation/pseudocosts.hpp"

#include "conftest.hpp"
#include "observation/unit-tests.hpp"

using namespace ecole;

TEST_CASE("Pseudocosts unit tests", "[unit][obs]") {
	observation::unit_tests(observation::Pseudocosts{});
}

TEST_CASE("Pseudocosts return pseudo costs array", "[obs]") {
	auto obs_func = observation::Pseudocosts{};
	auto model = get_model();
	obs_func.before_reset(model);
	advance_to_stage(model, SCIP_STAGE_SOLVING);
	auto const obs = obs_func.extract(model, false);

	REQUIRE(obs.has_value());
	auto const& costs = obs.value();
	REQUIRE(costs.size() == model.variables().size());

	// All branching candidates have a positive pseudocost
	for (auto* const var : model.lp_branch_cands()) {
		auto const var_index = static_cast<std::size_t>(SCIPvarGetProbindex(var));
		auto const pseudocost = costs[var_index];
		REQUIRE(!std::isnan(pseudocost));
		REQUIRE(pseudocost > 0);
	}
}
