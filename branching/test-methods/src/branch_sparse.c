#include <assert.h>
#include <string.h>
#include <math.h>

#include "branch_sparse.h"

#define BRANCHRULE_NAME            "sparse"
#define BRANCHRULE_DESC            "sparse branching rule"
#define BRANCHRULE_PRIORITY        50000
#define BRANCHRULE_MAXDEPTH        -1
#define BRANCHRULE_MAXBOUNDDIST    1.0

#define max(a,b)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b;       \
})

#define min(a,b)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b;       \
})


double safe_div(double x, double y) {
   return y == 0. ? 0. : x / y ;
}

double qb_div(double x, double y) {
   return y == 0. ? x : x / y ;
}

double safe_inv(double x) {
   return x != 0. ? 1. / x : 1.; 
}

double square(double x) { 
   return x * x; 
}

void rows_stats(SCIP_COL* col, int nvars, int* rows_nnz_nonfixed_arr, int* rows_deg_sum, int* rows_deg_min, int* rows_deg_max) {

   SCIP_ROW** rows;
   SCIP_ROW* row;
   int j;
   int row_lp_id, row_nnz_nonfixed;

   rows = SCIPcolGetRows(col);

   *rows_deg_sum = 0;
   *rows_deg_min = nvars; //max possible
   *rows_deg_max = 0; //min possible

   int lp_rowcount = SCIPcolGetNLPNonz(col);

   for (j = 0; j < lp_rowcount; j++)
   {
      row = rows[j];
      row_lp_id = SCIProwGetLPPos(row);
      //assert(row_lp_id >= 0);

      row_nnz_nonfixed = rows_nnz_nonfixed_arr[row_lp_id];
      *rows_deg_sum += row_nnz_nonfixed;
      *rows_deg_min = min(row_nnz_nonfixed, *rows_deg_min);
      *rows_deg_max = max(row_nnz_nonfixed, *rows_deg_max);

   }

   if (lp_rowcount == 0) {
      *rows_deg_min = 0;
      *rows_deg_max = 0;
   }

}

void rows_stddev(SCIP_COL* col, int* rows_nnz_nonfixed_arr, double* rows_deg_stdev, double mean) {
   SCIP_ROW** rows;
   SCIP_ROW* row;
   int row_nnz_nonfixed;
   int j, lp_rowcount;

   int row_lp_id;   
   rows = SCIPcolGetRows(col);

   lp_rowcount = SCIPcolGetNLPNonz(col);

   *rows_deg_stdev = 0;
   for (j = 0; j < lp_rowcount; j++)
   {
      row = rows[j];
      row_lp_id = SCIProwGetLPPos(row);

      row_nnz_nonfixed = rows_nnz_nonfixed_arr[row_lp_id];

      *rows_deg_stdev += (row_nnz_nonfixed - mean) * (row_nnz_nonfixed - mean);
   }

   *rows_deg_stdev = (lp_rowcount == 0 ? 0 : *rows_deg_stdev / lp_rowcount);

   *rows_deg_stdev = sqrt(*rows_deg_stdev);
}

void rows_pos_neg_coefficients_stats(SCIP_COL* col, double maxval, int* rows_pos_coefs_count, int* rows_neg_coefs_count, double* rows_pos_coefs_min, double* rows_pos_coefs_max, double* rows_pos_coefs_sum, double* rows_neg_coefs_min, double* rows_neg_coefs_max, double* rows_neg_coefs_sum)
{
   SCIP_Real* col_vals;

   int j, lp_rowcount;

   col_vals = SCIPcolGetVals(col);

   *rows_pos_coefs_sum = 0;
   *rows_neg_coefs_sum = 0;
   
   *rows_pos_coefs_min = maxval;
   *rows_pos_coefs_max = 0;

   *rows_neg_coefs_min = 0;
   *rows_neg_coefs_max = -1 * maxval;

   lp_rowcount = SCIPcolGetNLPNonz(col);

   *rows_pos_coefs_count = 0;
   *rows_neg_coefs_count = 0;

   for(j = 0; j < lp_rowcount; j++)
   {
      if (col_vals[j] > 0) {
         ++(*rows_pos_coefs_count);
         *rows_pos_coefs_sum += col_vals[j];
         *rows_pos_coefs_min = min(col_vals[j], *rows_pos_coefs_min);
         *rows_pos_coefs_max = max(col_vals[j], *rows_pos_coefs_max);
      }

      else if (col_vals[j] < 0) {
         ++(*rows_neg_coefs_count);
         *rows_neg_coefs_sum += col_vals[j];
         *rows_neg_coefs_min = min(col_vals[j], *rows_neg_coefs_min);
         *rows_neg_coefs_max = max(col_vals[j], *rows_neg_coefs_max);

      }
   }

   if (*rows_pos_coefs_count == 0) {
      *rows_pos_coefs_min = 0;
      *rows_pos_coefs_max = 0;
   }
   if (*rows_neg_coefs_count == 0) {
      *rows_neg_coefs_min = 0;
      *rows_neg_coefs_max = 0;
   }
}

void rows_pos_neg_coefficients_stddev(SCIP_COL* col, double* rows_pos_coefs_stdev, double rows_pos_coefs_mean, double* rows_neg_coefs_stdev, double rows_neg_coefs_mean) {

   SCIP_Real* col_vals;
   int j, lp_rowcount;

   col_vals = SCIPcolGetVals(col);

   *rows_pos_coefs_stdev = 0;
   *rows_neg_coefs_stdev = 0;

   lp_rowcount = SCIPcolGetNLPNonz(col);

   for(j = 0; j < lp_rowcount; j++)
   {
      if (col_vals[j] > 0) {
         *rows_pos_coefs_stdev += square(col_vals[j] - rows_pos_coefs_mean);
      }

      else if (col_vals[j] < 0) {
         *rows_neg_coefs_stdev += square(col_vals[j] - rows_neg_coefs_mean);
      }
   }
}

void set_min_max_for_ratios_constraint_coeffs_rhs(SCIP_COL* col, double* rows_rhs_arr, double* rows_lhs_arr, double* positive_rhs_ratio_min, double* positive_rhs_ratio_max, double* negative_rhs_ratio_min, double* negative_rhs_ratio_max)
{
   SCIP_Real* col_vals;
   SCIP_ROW** rows;
   SCIP_ROW* row;

   int j, row_lp_index;
   double rhs, lhs;
   double ratio_val;
   int lp_rowcount;

   col_vals = SCIPcolGetVals(col);

   *positive_rhs_ratio_min = 1.0;
   *positive_rhs_ratio_max = -1.0;

   *negative_rhs_ratio_min = 1.0;
   *negative_rhs_ratio_max = -1.0;

   rows = SCIPcolGetRows(col);
   
   lp_rowcount = SCIPcolGetNLPNonz(col);

   double col_val;

   for(j = 0; j < lp_rowcount; j++)
   {
      row = rows[j];

      row_lp_index = SCIProwGetLPPos(row);

      rhs = rows_rhs_arr[row_lp_index];
      lhs = -1 * rows_lhs_arr[row_lp_index];

      if  ( !isnan(rhs) ) {

         col_val = col_vals[j];
         ratio_val = safe_div( col_val, (fabs(col_val) + fabs(rhs)) );

         if (rhs >= 0) {
            *positive_rhs_ratio_min = min( *positive_rhs_ratio_min, ratio_val);
            *positive_rhs_ratio_max = max( *positive_rhs_ratio_max, ratio_val);
         }
         else {
            *negative_rhs_ratio_min = min( *negative_rhs_ratio_min, ratio_val );
            *negative_rhs_ratio_max = max( *negative_rhs_ratio_max, ratio_val );
         }
      }

      
      if  ( !isnan(lhs) ) {
         
         col_val = -col_vals[j];

         ratio_val = safe_div( col_val, (fabs(col_val) + fabs(lhs)) );

         if (lhs >= 0) {
            *positive_rhs_ratio_min = min( *positive_rhs_ratio_min, ratio_val );
            *positive_rhs_ratio_max = max( *positive_rhs_ratio_max, ratio_val );
         }

         else {
            *negative_rhs_ratio_min = min( *negative_rhs_ratio_min, ratio_val );
            *negative_rhs_ratio_max = max( *negative_rhs_ratio_max, ratio_val );
         }
      }
   }
}

void set_min_max_for_one_to_all_coefficient_ratios(SCIP_COL* col,
      double* rows_positive_sum_arr, double* rows_negative_sum_arr,
      double* positive_positive_ratio_max, double* positive_positive_ratio_min, 
      double* positive_negative_ratio_max, double* positive_negative_ratio_min,
      double* negative_positive_ratio_max, double* negative_positive_ratio_min,
      double* negative_negative_ratio_max, double* negative_negative_ratio_min)
{

   SCIP_Real* col_vals;
   SCIP_ROW** rows;
   SCIP_ROW* row;

   int i, lp_rowcount, row_lp_index;

   double val;
   
   double positive_ratio, negative_ratio;

   lp_rowcount = SCIPcolGetNLPNonz(col);
 
   rows = SCIPcolGetRows(col);

   col_vals = SCIPcolGetVals(col);

   *positive_positive_ratio_max = 0.;
   *positive_positive_ratio_min = 1.;
   *positive_negative_ratio_max = 0.;
   *positive_negative_ratio_min = 1.;
   *negative_positive_ratio_max = 0.;
   *negative_positive_ratio_min = 1.;
   *negative_negative_ratio_max = 0.;
   *negative_negative_ratio_min = 1.;


   for (i = 0; i < lp_rowcount; i++) {
      row = rows[i];
      row_lp_index = SCIProwGetLPPos(row);

      val = col_vals[i];

      if (val > 0) {

         positive_ratio =  val / rows_positive_sum_arr[row_lp_index];
         negative_ratio = val / (val - rows_negative_sum_arr[row_lp_index]);
         *positive_positive_ratio_max = max(*positive_positive_ratio_max, positive_ratio);
         *positive_positive_ratio_min = min(*positive_positive_ratio_min, positive_ratio);
			*positive_negative_ratio_max = max(*positive_negative_ratio_max, negative_ratio);
			*positive_negative_ratio_min = min(*positive_negative_ratio_min, negative_ratio);

      }

      else if (val < 0) {

			positive_ratio = val / (val - rows_positive_sum_arr[row_lp_index]);
			negative_ratio = val / rows_negative_sum_arr[row_lp_index];
         *negative_positive_ratio_max = max(*negative_positive_ratio_max, positive_ratio);
			*negative_positive_ratio_min = min(*negative_positive_ratio_min, positive_ratio);
			*negative_negative_ratio_max = max(*negative_negative_ratio_max, negative_ratio);
			*negative_negative_ratio_min = min(*negative_negative_ratio_min, negative_ratio);

      }
   }

}


void set_row_weights(SCIP* scip, int* rows_is_active_arr, double* weights, double* row_absolute_sum, double* row_absolute_sum_candidates)  {

   SCIP_ROW** rows;
   SCIP_ROW* row;
   int nlprows, i;

   nlprows = SCIPgetNLPRows(scip);
   rows = SCIPgetLPRows(scip);

   for (i = 0; i < nlprows; i++) {
      row = rows[i];

      if ( rows_is_active_arr[i] ) {

			weights[i * 4] = 1.;
			weights[i * 4 + 1] = safe_inv(row_absolute_sum[i]);
			weights[i * 4 + 2] = safe_inv(row_absolute_sum_candidates[i]);
			weights[i * 4 + 3] = fabs(SCIProwGetDualsol(row));
      }
      /*
      else {
         weights[i * 4] = 0.;
         weights[i * 4 + 1] = 0.;
         weights[i * 4 + 2] = 0.;
         weights[i * 4 + 3] = 0.;
      }
      */
   }
}

void active_rows_weighted_coefficients_stats(SCIP* scip, SCIP_COL* col, int* rows_is_active_arr,
   double* weights, int* n_active_rows,
   double* active_rows_weight1_count, double* active_rows_weight1_sum, double* active_rows_weight1_min, double* active_rows_weight1_max, 
   double* active_rows_weight2_count, double* active_rows_weight2_sum, double* active_rows_weight2_min, double* active_rows_weight2_max, 
   double* active_rows_weight3_count, double* active_rows_weight3_sum, double* active_rows_weight3_min, double* active_rows_weight3_max, 
   double* active_rows_weight4_count, double* active_rows_weight4_sum, double* active_rows_weight4_min, double* active_rows_weight4_max
   
   )

{

   SCIP_Real* col_vals;
   SCIP_ROW* row;
   SCIP_ROW** rows;
   int j, lp_rowcount, row_lp_index;

   double weighted_abs_coef;

   *active_rows_weight1_sum = 0.0;
   *active_rows_weight2_sum = 0.0;
   *active_rows_weight3_sum = 0.0;
   *active_rows_weight4_sum = 0.0;

   *active_rows_weight1_count = 0.0;
   *active_rows_weight2_count = 0.0;
   *active_rows_weight3_count = 0.0;
   *active_rows_weight4_count = 0.0;

   //Nonnegative weights

   *active_rows_weight1_min = SCIPinfinity(scip);
   *active_rows_weight1_max = 0;

   *active_rows_weight2_min = SCIPinfinity(scip);
   *active_rows_weight2_max = 0;

   *active_rows_weight3_min = SCIPinfinity(scip);
   *active_rows_weight3_max = 0;

   *active_rows_weight4_min = SCIPinfinity(scip);
   *active_rows_weight4_max = 0;

   *n_active_rows = 0;

   col_vals = SCIPcolGetVals(col);
   lp_rowcount = SCIPcolGetNLPNonz(col);

   rows = SCIPcolGetRows(col);

   for(j = 0; j < lp_rowcount; j++) {
      row = rows[j];
      row_lp_index = SCIProwGetLPPos(row);
      //assert(row_lp_index >= 0);
      if ( rows_is_active_arr[row_lp_index] ) {

         weighted_abs_coef = weights[4 * row_lp_index] * fabs(col_vals[j]);
         (*active_rows_weight1_sum) += weighted_abs_coef;
         *active_rows_weight1_min = min(*active_rows_weight1_min, weighted_abs_coef);
         *active_rows_weight1_max = max(*active_rows_weight1_max, weighted_abs_coef);
         (*active_rows_weight1_count) += weights[4 * row_lp_index];


         weighted_abs_coef = weights[4 * row_lp_index + 1] * fabs(col_vals[j]);
         (*active_rows_weight2_sum) += weighted_abs_coef;
         *active_rows_weight2_min = min(*active_rows_weight2_min, weighted_abs_coef);
         *active_rows_weight2_max = max(*active_rows_weight2_max, weighted_abs_coef);
         (*active_rows_weight2_count) += weights[4 * row_lp_index + 1];


         weighted_abs_coef = weights[4 * row_lp_index + 2] * fabs(col_vals[j]);
         (*active_rows_weight3_sum) += weighted_abs_coef;
         *active_rows_weight3_min = min(*active_rows_weight3_min, weighted_abs_coef);
         *active_rows_weight3_max = max(*active_rows_weight3_max, weighted_abs_coef);
         (*active_rows_weight3_count) += weights[4 * row_lp_index + 2];


         weighted_abs_coef = weights[4 * row_lp_index + 3] * fabs(col_vals[j]);
         (*active_rows_weight4_sum) += weighted_abs_coef;
         *active_rows_weight4_min = min(*active_rows_weight4_min, weighted_abs_coef);
         *active_rows_weight4_max = max(*active_rows_weight4_max, weighted_abs_coef);
         (*active_rows_weight4_count) += weights[4 * row_lp_index + 3];

         (*n_active_rows) ++;
      }
   }
}

void active_rows_weighted_coefficients_stddev(SCIP_COL* col, int* rows_is_active_arr,
   double* weights, int n_active_rows,
   double* active_rows_weight1_stddev, double active_rows_weight1_mean,
   double* active_rows_weight2_stddev, double active_rows_weight2_mean,
   double* active_rows_weight3_stddev, double active_rows_weight3_mean,
   double* active_rows_weight4_stddev, double active_rows_weight4_mean)
{
   SCIP_Real* col_vals;
   SCIP_ROW* row;
   SCIP_ROW** rows;
   int j, lp_rowcount, row_lp_index;

   double weighted_abs_coef;

   col_vals = SCIPcolGetVals(col);
   lp_rowcount = SCIPcolGetNLPNonz(col);

   rows = SCIPcolGetRows(col);

   *active_rows_weight1_stddev = 0.0;
   *active_rows_weight2_stddev = 0.0;
   *active_rows_weight3_stddev = 0.0;
   *active_rows_weight4_stddev = 0.0;

   for(j = 0; j < lp_rowcount; j++) {

      row = rows[j];
      row_lp_index = SCIProwGetLPPos(row); // used to extract weights
      //assert(row_lp_index >= 0);
      if ( rows_is_active_arr[row_lp_index] ) {

         weighted_abs_coef = weights[4 * row_lp_index] * fabs(col_vals[j]);
         *active_rows_weight1_stddev += square(weighted_abs_coef - active_rows_weight1_mean);

         weighted_abs_coef = weights[4 * row_lp_index + 1] * fabs(col_vals[j]);
         *active_rows_weight2_stddev += square(weighted_abs_coef - active_rows_weight2_mean);

         weighted_abs_coef = weights[4 * row_lp_index + 2] * fabs(col_vals[j]);
         *active_rows_weight3_stddev += square(weighted_abs_coef - active_rows_weight3_mean);

         weighted_abs_coef = weights[4 * row_lp_index + 3] * fabs(col_vals[j]);
         *active_rows_weight4_stddev += square(weighted_abs_coef - active_rows_weight4_mean);

      }
   }

   *active_rows_weight1_stddev = sqrt(safe_div(*active_rows_weight1_stddev, n_active_rows));
   *active_rows_weight2_stddev = sqrt(safe_div(*active_rows_weight2_stddev, n_active_rows));
   *active_rows_weight3_stddev = sqrt(safe_div(*active_rows_weight3_stddev, n_active_rows));
   *active_rows_weight4_stddev = sqrt(safe_div(*active_rows_weight4_stddev, n_active_rows));

}


double obj_l2_norm(SCIP* scip) {
	double norm = SCIPgetObjNorm(scip);
	return norm > 0 ? norm : 1.;
}

double row_l2_norm(SCIP_ROW* row) {
	double norm = SCIProwGetNorm(row);
	return norm > 0 ? norm : 1.;
}


int is_prim_sol_at_lb(SCIP* scip, SCIP_COL* col) {
	double lb_val = SCIPcolGetLb(col);
	if (!SCIPisInfinity(scip, fabs(lb_val))) {
		return SCIPisEQ(scip, SCIPcolGetPrimsol(col), lb_val);
	}
	return 0;
}

int is_prim_sol_at_ub(SCIP* scip, SCIP_COL* col) {
	double ub_val = SCIPcolGetUb(col);
	if (!SCIPisInfinity(scip, fabs(ub_val))) {
		return SCIPisEQ(scip, SCIPcolGetPrimsol(col), ub_val);
	}
	return 0;
}

double best_sol_val(SCIP* scip, SCIP_VAR* var) {
	SCIP_SOL* sol = SCIPgetBestSol(scip);
	if (sol != NULL) {
		return SCIPgetSolVal(scip, sol, var);
	}
	return 0; // NAN
}

double avg_sol(SCIP* scip, SCIP_VAR* var) {
	if (SCIPgetBestSol(scip) != NULL) {
		return SCIPvarGetAvgSol(var);
	}
	return 0; // NAN?
}


double get_unshifted_lhs(SCIP* scip, SCIP_ROW* row) {
	double lhs_val = SCIProwGetLhs(row);
	if ( SCIPisInfinity(scip, fabs(lhs_val)) ) {
		return NAN; //NAN
	}
	return lhs_val - SCIProwGetConstant(row);
}

double get_unshifted_rhs(SCIP* scip, SCIP_ROW* row) {
	double rhs_val = SCIProwGetRhs(row);
	if ( SCIPisInfinity(scip, fabs(rhs_val)) ) {
		return NAN; //NAN
	}
	return rhs_val - SCIProwGetConstant(row);
}

double obj_cos_sim(SCIP* scip, SCIP_ROW* row) {
	double norm_prod = SCIProwGetNorm(row) * SCIPgetObjNorm(scip);
	if (SCIPisPositive(scip, norm_prod)) {
		return safe_div(row->objprod, norm_prod);
	}
	return 0.;
}

int is_at_rhs(SCIP* scip, SCIP_ROW* row) {
	double activity = SCIPgetRowLPActivity(scip, row);
	double rhs_val = SCIProwGetRhs(row);
	return SCIPisEQ(scip, activity, rhs_val);
}

int is_at_lhs(SCIP* scip, SCIP_ROW* row) {
	double activity = SCIPgetRowLPActivity(scip, row);
	double lhs_val = SCIProwGetLhs(row);
	return SCIPisEQ(scip, activity, lhs_val);
}

static
SCIP_DECL_BRANCHEXECLP(branchExeclpSparse)
{  

   SCIP_VAR** lpcands;
   SCIP_VAR** vars;
   SCIP_VAR* var;
   SCIP_Real* lpcandsfrac;
   SCIP_Real* col_vals;
   int nlpcands;
   int i, j;
   double rows_deg_stddev;
   int rows_deg_sum, rows_deg_min, rows_deg_max;
   int nvars = SCIPgetNVars(scip);

   int rows_pos_coefs_count;
   int rows_neg_coefs_count;

   double rows_pos_coefs_max, rows_pos_coefs_min, rows_neg_coefs_max, rows_neg_coefs_min;
   double rows_pos_coefs_sum, rows_neg_coefs_sum;
   double rows_pos_coefs_stddev, rows_neg_coefs_stddev;

   int n_infeasibles_up, n_infeasibles_down, n_branchings_up, n_branchings_down;
   
   double positive_rhs_ratio_min, positive_rhs_ratio_max, negative_rhs_ratio_min, negative_rhs_ratio_max;

   double positive_positive_ratio_max, positive_positive_ratio_min, positive_negative_ratio_max, positive_negative_ratio_min;
   double negative_positive_ratio_max, negative_positive_ratio_min, negative_negative_ratio_max, negative_negative_ratio_min;


   double epsilon = 1e-5;
   double wpu_approx;
   double wpl_approx;

   double active_rows_weight1_count, active_rows_weight2_count, active_rows_weight3_count, active_rows_weight4_count;
   double active_rows_weight1_sum, active_rows_weight2_sum, active_rows_weight3_sum, active_rows_weight4_sum;
   double active_rows_weight1_min, active_rows_weight2_min, active_rows_weight3_min, active_rows_weight4_min;
   double active_rows_weight1_max, active_rows_weight2_max, active_rows_weight3_max, active_rows_weight4_max;
   double active_rows_weight1_mean, active_rows_weight2_mean, active_rows_weight3_mean, active_rows_weight4_mean;
   double active_rows_weight1_stddev, active_rows_weight2_stddev, active_rows_weight3_stddev, active_rows_weight4_stddev;

   int n_active_rows;

   int nlps;

   double edge_sum, edge_min, edge_max, edge_val;
   double bias_sum, bias_min, bias_max, bias_val;
   double obj_cos_sim_sum, obj_cos_sim_min, obj_cos_sim_max, obj_cos_sim_val;
   double is_tight_sum, is_tight_min, is_tight_max, is_tight_val;
   double dual_solution_sum, dual_solution_min, dual_solution_max, dual_solution_val;
   double scaled_age_sum, scaled_age_min, scaled_age_max, scaled_age_val;

   int row_counter;
	double row_norm;

   SCIP_ROW** rows;
   SCIP_ROW* row;

   int nrows = SCIPgetNLPRows(scip);
   rows = SCIPgetLPRows(scip);

   double activity, rhs, lhs;
   
   double lb, ub;

   SCIP_COL** cols;
   SCIP_COL* col;
   int ncols = SCIPgetNLPCols(scip);

   cols = SCIPgetLPCols(scip);

   if (nrows > row_abs_sum_size) {
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &row_weights, 4 * row_abs_sum_size, 4 * nrows) );

      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &row_abs_sum, row_abs_sum_size, nrows) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &row_abs_sum_candidates, row_abs_sum_size, nrows) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &rows_positive_sum, row_abs_sum_size, nrows) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &rows_negative_sum, row_abs_sum_size, nrows) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &rows_rhs, row_abs_sum_size, nrows) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &rows_lhs, row_abs_sum_size, nrows) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &rows_nnz_nonfixed, row_abs_sum_size, nrows) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &rows_reduced_norm, row_abs_sum_size, nrows) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &rows_reduced_obj_cos_sim, row_abs_sum_size, nrows) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &rows_is_active, row_abs_sum_size, nrows) );
      row_abs_sum_size = nrows;
   }

   double reduced_obj_norm = 0.0;

   double sol_val;

   for(i = 0; i < ncols; i++) {
      col = cols[i];
      var = SCIPcolGetVar(col);
      is_candidate[i] = 0;
      sol_val = SCIPvarGetLPSol(var);
      lb = SCIPcolGetLb(col);
      ub = SCIPcolGetUb(col);

      if (ub < lb + 0.5) //fixed
      {
         if (ub < 0.5) // fixed to 0
            columns_fixed_or_not[i] = 0;
         else // fixed to 1
            columns_fixed_or_not[i] = 1;
      }
      else { //not fixed
         columns_fixed_or_not[i] = -1;
         reduced_obj_norm += (SCIPcolGetObj(col) * SCIPcolGetObj(col));

         if ( !SCIPisIntegral(scip, sol_val) )
            is_candidate[i] = 1;
      }
   }

   reduced_obj_norm = sqrt(reduced_obj_norm);
   
   int row_ncols;
   int col_lp_id, row_lp_id;
   double reduced_sum;
   SCIP_COL** row_cols;

   double reduced_sum_square;
   double col_obj;
   double inner_product;
   double* row_nnz_values;
   double* row_vals;

   for(i = 0; i < nrows; i++) {
      row = rows[i];
      inner_product = 0.0;
      row_lp_id = SCIProwGetLPPos(row);

      row_cols = SCIProwGetCols(row);
      row_ncols = SCIProwGetNLPNonz(row);

      row_nnz_values = SCIProwGetVals(row);

      row_abs_sum[i] = 0.;
      row_abs_sum_candidates[i] = 0.;

      row_vals = SCIProwGetVals(row);
      
      rows_positive_sum[i] = 0.0;
      rows_negative_sum[i] = 0.0;

      reduced_sum = 0;
      reduced_sum_square = 0;
      rows_nnz_nonfixed[row_lp_id] = 0;

      activity = SCIPgetRowActivity(scip, row);
      lhs = SCIProwGetLhs(row);
      rhs = SCIProwGetRhs(row);
      
      rows_is_active[row_lp_id] = ( SCIPisEQ(scip, activity, rhs) || SCIPisEQ(scip, activity, lhs) );

      if( rows_is_active[row_lp_id] ) {

         for(j = 0; j < row_ncols; j++) {
            col_obj = SCIPcolGetObj(row_cols[j]);
            col_lp_id = SCIPcolGetLPPos(row_cols[j]);

            if (columns_fixed_or_not[col_lp_id] == -1) //nonfixed column
				{
               rows_nnz_nonfixed[row_lp_id] ++;
               
               reduced_sum_square += (row_nnz_values[j] * row_nnz_values[j]);

               inner_product += (row_nnz_values[j] * col_obj);
               
               if (row_vals[j] > 0) 
				      rows_positive_sum[i] += row_vals[j];
			      else
				      rows_negative_sum[i] += row_vals[j];

               row_abs_sum[i] += fabs(row_vals[j]);

               if ( is_candidate[col_lp_id] ) 
                  row_abs_sum_candidates[i] += fabs(row_vals[j]);
            }

            else if (columns_fixed_or_not[col_lp_id] == 1) {
               reduced_sum += row_nnz_values[j];
            }
         }
      }

      else {
         for(j = 0; j < row_ncols; j++) {
            col_obj = SCIPcolGetObj(row_cols[j]);
            col_lp_id = SCIPcolGetLPPos(row_cols[j]);

            if (columns_fixed_or_not[col_lp_id] == -1)  //nonfixed column
				{
               rows_nnz_nonfixed[row_lp_id] ++;
               
               reduced_sum_square += (row_nnz_values[j] * row_nnz_values[j]);

               inner_product += (row_nnz_values[j] * col_obj);
               
               if (row_vals[j] > 0) 
				      rows_positive_sum[i] += row_vals[j];
			      else
				      rows_negative_sum[i] += row_vals[j];
            
            }

            else if (columns_fixed_or_not[col_lp_id] == 1) {
               reduced_sum += row_nnz_values[j];
            }
         }
      }

      if ( !isnan( get_unshifted_rhs(scip, row) ))
         rows_rhs[row_lp_id] = get_unshifted_rhs(scip, row) - reduced_sum;
      else
         rows_rhs[row_lp_id] = NAN;
      
      if ( !isnan( get_unshifted_lhs(scip, row)))
         rows_lhs[row_lp_id] = get_unshifted_lhs(scip, row) - reduced_sum; 
      else
         rows_lhs[row_lp_id] = NAN;

      rows_reduced_norm[row_lp_id] = sqrt(reduced_sum_square);
      rows_reduced_obj_cos_sim[row_lp_id] = safe_div(inner_product, rows_reduced_norm[row_lp_id] * reduced_obj_norm);

      row_abs_sum[i] /= rows_reduced_norm[i];
      row_abs_sum_candidates[i] /= rows_reduced_norm[i];

   }

   nlps = SCIPgetNLPs(scip);

   assert(scip != NULL);
   assert(branchrule != NULL);
   assert(strcmp(SCIPbranchruleGetName(branchrule), BRANCHRULE_NAME) == 0);
   assert(result != NULL);

   if( SCIPgetLPSolstat(scip) != SCIP_LPSOLSTAT_OPTIMAL )
   {
      *result = SCIP_DIDNOTRUN;
      SCIPdebugMsg(scip, "Could not apply the sparse branching, as the current LP was not solved to optimality.\n");

      return SCIP_OKAY;
   }

   SCIP_CALL( SCIPgetLPBranchCands(scip, &lpcands, NULL, &lpcandsfrac, NULL, &nlpcands, NULL) );

   vars = SCIPgetVars(scip);

   assert(nlpcands > 0);
   assert(nlpcands <= nvars);

   if( SCIPgetDepth(scip) == 0 ) { // Extract root node features for all variables, including noncandidates

      for (i = 0; i < nvars; i++) {
 
         var = vars[i];

         col = SCIPvarGetCol(var);

         static_features[i * n_static_features] = SCIPcolGetObj(col); // K Feature 0: obj_coef
         static_features[i * n_static_features + 1] = max(SCIPcolGetObj(col), 0); // K Feature 1: obj_coef_pos_part
         static_features[i * n_static_features + 2] = min(SCIPcolGetObj(col), 0); // K Feature 2: obj_coef_neg_part
         static_features[i * n_static_features + 3] = SCIPcolGetNLPNonz(col); // K Feature 3: n_rows

         rows_stats(col, nvars, rows_nnz_nonfixed, &rows_deg_sum, &rows_deg_min, &rows_deg_max);

         static_features[i * n_static_features + 4] = safe_div( (double) rows_deg_sum, (double) SCIPcolGetNLPNonz(col) ); // K Feature 4: rows_deg_mean

         rows_stddev(col, rows_nnz_nonfixed, &rows_deg_stddev, static_features[i * n_static_features + 4]);

         static_features[i * n_static_features + 5] = rows_deg_stddev; // K Feature 5: rows_deg_stddev
         static_features[i * n_static_features + 6] = (double) rows_deg_min; // K Feature 6: rows_deg_min
         static_features[i * n_static_features + 7] = (double) rows_deg_max; // K Feature 7: rows_deg_max

         rows_pos_neg_coefficients_stats(col, SCIPinfinity(scip), &rows_pos_coefs_count, &rows_neg_coefs_count, &rows_pos_coefs_min, &rows_pos_coefs_max, &rows_pos_coefs_sum, &rows_neg_coefs_min, &rows_neg_coefs_max, &rows_neg_coefs_sum);

         static_features[i * n_static_features + 8] = (double) rows_pos_coefs_count; // K Feature 8: rows_pos_coefs_count
         static_features[i * n_static_features + 9] = safe_div(rows_pos_coefs_sum, (double) rows_pos_coefs_count); // K Feature 9: rows_pos_coefs_mean
         
         static_features[i * n_static_features + 11] = rows_pos_coefs_min; // K Feature 11: rows_pos_coefs_min
         static_features[i * n_static_features + 12] = rows_pos_coefs_max; // K Feature 12: rows_pos_coefs_max

         static_features[i * n_static_features + 13] = (double) rows_neg_coefs_count; // K Feature 13: rows_neg_coefs_count
         static_features[i * n_static_features + 14] = safe_div(rows_neg_coefs_sum, (double) rows_neg_coefs_count); // K Feature 14: rows_neg_coefs_mean
         
         static_features[i * n_static_features + 16] = rows_neg_coefs_min; // K Feature 16: rows_neg_coefs_min
         static_features[i * n_static_features + 17] = rows_neg_coefs_max; // K Feature 17: rows_neg_coefs_max

         rows_pos_neg_coefficients_stddev(col, &rows_pos_coefs_stddev, static_features[i * n_static_features + 9], &rows_neg_coefs_stddev, static_features[i * n_static_features + 14]);
         
         static_features[i * n_static_features + 10] = sqrt(safe_div(  rows_pos_coefs_stddev, (double) rows_pos_coefs_count) ); // K Feature 10: rows_pos_coefs_stddev
         static_features[i * n_static_features + 15] = sqrt(safe_div(  rows_neg_coefs_stddev, (double) rows_neg_coefs_count) ); // K Feature 15: rows_neg_coefs_stddev

         static_features[i * n_static_features + 18] = SCIPcolGetObj(col) / reduced_obj_norm; // G Feature 0: objective
         static_features[i * n_static_features + 19] = 0.0; // G Feature 1: is_type_binary
         static_features[i * n_static_features + 20] = 0.0; // G Feature 2: is_type_integer
         static_features[i * n_static_features + 21] = 0.0; // G Feature 3: is_type_implicit_integer
         static_features[i * n_static_features + 22] = 0.0; // G Feature 4: is_type_continuous

         switch (SCIPvarGetType(var)) {
	         case SCIP_VARTYPE_BINARY:
		         static_features[i * n_static_features + 19] = 1.;
		         break;
	         case SCIP_VARTYPE_INTEGER:
               static_features[i * n_static_features + 20] = 1.;
		         break;
	         case SCIP_VARTYPE_IMPLINT:
		         static_features[i * n_static_features + 21] = 1.;
		         break;
	         case SCIP_VARTYPE_CONTINUOUS:
               static_features[i * n_static_features + 22] = 1.;
               break;
            default:
               break;
         }
      }
   }

   set_row_weights(scip, rows_is_active, row_weights, row_abs_sum, row_abs_sum_candidates);
   int index;

   // Extract dynamic features for branching candidates only

   for(i = 0; i < nlpcands; i++) {

      var = lpcands[i];
      
      index = SCIPvarGetProbindex(var);
      
      for (j = 0; j < n_K_static; j++)
         features[i * n_features + j] = static_features[index * n_static_features + j];

      scores[i] = 0;

      col = SCIPvarGetCol(var);

      double solval = SCIPcolGetPrimsol(col);

      double floor_dist = SCIPfeasFrac(scip, solval);
      double ceil_dist = 1.0 - floor_dist;

      features[i * n_features + 18] = min(floor_dist, ceil_dist); // K Feature 18: slack
      features[i * n_features + 19] = ceil_dist; // K Feature 19: ceil_dist

      features[i * n_features + 20] = ceil_dist * SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_UPWARDS); // K Feature 20: pseudocost_up
      features[i * n_features + 21] = floor_dist * SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_DOWNWARDS); // K Feature 21: pseudocost_down

      wpu_approx = max(features[i * n_features + 20], epsilon);
	   wpl_approx = max(features[i * n_features + 21], epsilon);

      double min_wp = min(wpu_approx, wpl_approx);
      double max_wp = max(wpu_approx, wpl_approx);

      features[i * n_features + 22] = max_wp != 0. ? min_wp / max_wp : 0.; // K Feature 22: pseudocost_ratio
      
      features[i * n_features + 23] = features[i * n_features + 20] + features[i * n_features + 21]; // K Feature 23: pseudocost_sum
      features[i * n_features + 24] = features[i * n_features + 20] * features[i * n_features + 21]; // K Feature 24: pseudocost_product

      n_infeasibles_up = SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_UPWARDS); 
      n_infeasibles_down = SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_DOWNWARDS); 
      
      n_branchings_up = SCIPvarGetNBranchings(var, SCIP_BRANCHDIR_UPWARDS);
      n_branchings_down = SCIPvarGetNBranchings(var, SCIP_BRANCHDIR_DOWNWARDS);

      features[i * n_features + 25] = (double) n_infeasibles_up; // K Feature 25: n_cutoff_up
      features[i * n_features + 26] = (double) n_infeasibles_down; // K Feature 26: n_cutoff_down

      features[i * n_features + 27] = safe_div( (double) n_infeasibles_up, (double) n_branchings_up); // K Feature 27: n_cutoff_up_ratio
      features[i * n_features + 28] = safe_div( (double) n_infeasibles_down, (double) n_branchings_down); // K Feature 28: n_cutoff_down_ratio

      rows_stats(col, nvars, rows_nnz_nonfixed, &rows_deg_sum, &rows_deg_min, &rows_deg_max);

      features[i * n_features + 29] = safe_div( rows_deg_sum, SCIPcolGetNLPNonz(col) ); // K Feature 29: rows_dynamic_deg_mean

      rows_stddev(col, rows_nnz_nonfixed, &rows_deg_stddev, features[i * n_features + 29]);

      features[i * n_features + 30] = rows_deg_stddev; // K Feature 30: rows_dynamic_deg_stddev
      features[i * n_features + 31] = rows_deg_min; // K Feature 31: rows_dynamic_deg_min
      features[i * n_features + 32] = rows_deg_max; // K Feature 32: rows_dynamic_deg_max

      features[i * n_features + 33] = safe_div(features[i * n_features + 29], features[i * n_features + 4] + features[i * n_features + 29]) ; // K Feature 33: rows_dynamic_deg_mean_ratio
      features[i * n_features + 34] = safe_div(features[i * n_features + 31], features[i * n_features + 6] + features[i * n_features + 31]) ; // K Feature 34: rows_dynamic_deg_min_ratio
      features[i * n_features + 35] = safe_div(features[i * n_features + 32], features[i * n_features + 7] + features[i * n_features + 32]) ; // K Feature 35: rows_dynamic_deg_max_ratio

      set_min_max_for_ratios_constraint_coeffs_rhs(col, rows_rhs, rows_lhs, &positive_rhs_ratio_min, &positive_rhs_ratio_max, &negative_rhs_ratio_min, &negative_rhs_ratio_max);
      
      features[i * n_features + 36] = positive_rhs_ratio_min; // K Feature 36: coef_pos_rhs_ratio_min
      features[i * n_features + 37] = positive_rhs_ratio_max; // K Feature 37: coef_pos_rhs_ratio_max
      features[i * n_features + 38] = negative_rhs_ratio_min; // K Feature 38: coef_neg_rhs_ratio_min
      features[i * n_features + 39] = negative_rhs_ratio_max; // K Feature 39: coef_neg_rhs_ratio_max

      set_min_max_for_one_to_all_coefficient_ratios(col, rows_positive_sum, rows_negative_sum, &positive_positive_ratio_max, &positive_positive_ratio_min, &positive_negative_ratio_max, &positive_negative_ratio_min,
         &negative_positive_ratio_max, &negative_positive_ratio_min, &negative_negative_ratio_max, &negative_negative_ratio_min);

      features[i * n_features + 40] = positive_positive_ratio_min; // K Feature 40: pos_coef_pos_coef_ratio_min
      features[i * n_features + 41] = positive_positive_ratio_max; // K Feature 41: pos_coef_pos_coef_ratio_max
      features[i * n_features + 42] = positive_negative_ratio_min; // K Feature 42: pos_coef_neg_coef_ratio_min
      features[i * n_features + 43] = positive_negative_ratio_max; // K Feature 43: pos_coef_neg_coef_ratio_max
      features[i * n_features + 44] = negative_positive_ratio_min; // K Feature 44: neg_coef_pos_coef_ratio_min
      features[i * n_features + 45] = negative_positive_ratio_max; // K Feature 45: neg_coef_pos_coef_ratio_max
      features[i * n_features + 46] = negative_negative_ratio_min; // K Feature 46: neg_coef_neg_coef_ratio_min
      features[i * n_features + 47] = negative_negative_ratio_max; // K Feature 47: neg_coef_neg_coef_ratio_max
      
      active_rows_weighted_coefficients_stats(scip, col, rows_is_active, row_weights, &n_active_rows,
         &active_rows_weight1_count, &active_rows_weight1_sum, &active_rows_weight1_min, &active_rows_weight1_max,
         &active_rows_weight2_count, &active_rows_weight2_sum, &active_rows_weight2_min, &active_rows_weight2_max,
         &active_rows_weight3_count, &active_rows_weight3_sum, &active_rows_weight3_min, &active_rows_weight3_max,
         &active_rows_weight4_count, &active_rows_weight4_sum, &active_rows_weight4_min, &active_rows_weight4_max
         );

      active_rows_weight1_mean = safe_div(active_rows_weight1_sum, n_active_rows);
      active_rows_weight2_mean = safe_div(active_rows_weight2_sum, n_active_rows);
      active_rows_weight3_mean = safe_div(active_rows_weight3_sum, n_active_rows);
      active_rows_weight4_mean = safe_div(active_rows_weight4_sum, n_active_rows);

      active_rows_weighted_coefficients_stddev(col, rows_is_active, row_weights, n_active_rows,
         &active_rows_weight1_stddev, active_rows_weight1_mean,
         &active_rows_weight2_stddev, active_rows_weight2_mean,
         &active_rows_weight3_stddev, active_rows_weight3_mean,
         &active_rows_weight4_stddev, active_rows_weight4_mean);

      features[i * n_features + 48] = active_rows_weight1_count; // K Feature 48: active_coef_weight1_count
      features[i * n_features + 49] = active_rows_weight1_sum; // K Feature 49: active_coef_weight1_sum
      features[i * n_features + 50] = active_rows_weight1_mean; // K Feature 50: active_coef_weight1_mean
      features[i * n_features + 51] = active_rows_weight1_stddev; // K Feature 51: active_coef_weight1_stddev
      features[i * n_features + 52] = active_rows_weight1_min; // K Feature 52: active_coef_weight1_min
      features[i * n_features + 53] = active_rows_weight1_max; // K Feature 53: active_coef_weight1_max

      features[i * n_features + 54] = active_rows_weight2_count; // K Feature 54: active_coef_weight2_count
      features[i * n_features + 55] = active_rows_weight2_sum; // K Feature 55: active_coef_weight2_sum
      features[i * n_features + 56] = active_rows_weight2_mean; // K Feature 56: active_coef_weight2_mean
      features[i * n_features + 57] = active_rows_weight2_stddev; // K Feature 57: active_coef_weight2_stddev
      features[i * n_features + 58] = active_rows_weight2_min; // K Feature 58: active_coef_weight2_min
      features[i * n_features + 59] = active_rows_weight2_max; // K Feature 59: active_coef_weight2_max
      
      features[i * n_features + 60] = active_rows_weight3_count; // K Feature 60: active_coef_weight3_count
      features[i * n_features + 61] = active_rows_weight3_sum; // K Feature 61: active_coef_weight3_sum
      features[i * n_features + 62] = active_rows_weight3_mean; // K Feature 62: active_coef_weight3_mean
      features[i * n_features + 63] = active_rows_weight3_stddev; // K Feature 63: active_coef_weight3_stddev
      features[i * n_features + 64] = active_rows_weight3_min; // K Feature 64: active_coef_weight3_min
      features[i * n_features + 65] = active_rows_weight3_max; // K Feature 65: active_coef_weight3_max
      
      features[i * n_features + 66] = active_rows_weight4_count; // K Feature 66: active_coef_weight4_count
      features[i * n_features + 67] = active_rows_weight4_sum; // K Feature 67: active_coef_weight4_sum
      features[i * n_features + 68] = active_rows_weight4_mean; // K Feature 68: active_coef_weight4_mean
      features[i * n_features + 69] = active_rows_weight4_stddev; // K Feature 69: active_coef_weight4_stddev
      features[i * n_features + 70] = active_rows_weight4_min; // K Feature 70: active_coef_weight4_min
      features[i * n_features + 71] = active_rows_weight4_max;  // K Feature 71: active_coef_weight4_max

      for (j = 0; j < n_G_static; j++ )
         features[i * n_features + j + 72] = static_features[index * n_static_features + j + 18];

      double lb_val = SCIPcolGetLb(col);
      double ub_val = SCIPcolGetUb(col);

      if ( SCIPisInfinity(scip, fabs(lb_val)) )
         features[i * n_features + 77] = 0.0; // G Feature 5: has_lower_bound
      else 
         features[i * n_features + 77] = 1.0;

      if ( SCIPisInfinity(scip, fabs(ub_val)) )
         features[i * n_features + 78] = 0.0; // G Feature 6: has_upper_bound
      else 
         features[i * n_features + 78] = 1.0;

      
      features[i * n_features + 79] = safe_div(SCIPgetVarRedcost(scip, var), reduced_obj_norm); // G Feature 7: normed_reduced_cost
      features[i * n_features + 80] = SCIPvarGetLPSol(var); // G Feature 8: solution_value
      features[i * n_features + 81] = SCIPfeasFrac( scip, SCIPvarGetLPSol(var) ); // G Feature 9: solution_frac
      features[i * n_features + 82] = is_prim_sol_at_lb(scip, col); // G Feature 10:is_solution_at_lower_bound
      features[i * n_features + 83] = is_prim_sol_at_ub(scip, col); // G Feature 11: is_solution_at_upper_bound
      features[i * n_features + 84] = (double) SCIPcolGetAge(col) / (nlps + 5.0); // G Feature 12: scaled_age
      features[i * n_features + 85] = best_sol_val(scip, var); // G Feature 13: incumbent_value
      features[i * n_features + 86] = avg_sol(scip, var); // G Feature 14: average_incumbent_value

      //Basis
      features[i * n_features + 87] = 0.; // G Feature 15: is_basis_lower
      features[i * n_features + 88] = 0.; // G Feature 16: is_basis_basic
      features[i * n_features + 89] = 0.; // G Feature 17: is_basis_upper
      features[i * n_features + 90] = 0.; // G Feature 18: is_basis_zero

      switch (SCIPcolGetBasisStatus(col)) {
	      case SCIP_BASESTAT_LOWER:
		      features[i * n_features + 87] = 1.;
		      break;
         case SCIP_BASESTAT_BASIC:
            features[i * n_features + 88] = 1.;
            break;
         case SCIP_BASESTAT_UPPER:
            features[i * n_features + 89] = 1.;
            break;
         case SCIP_BASESTAT_ZERO:
            features[i * n_features + 90] = 1.;
            break; 
	   }

      features[i * n_features + 91] = min( SCIPvarGetLPSol(var) - SCIPfeasFloor(scip, SCIPvarGetLPSol(var)), SCIPfeasCeil(scip, SCIPvarGetLPSol(var)) - SCIPvarGetLPSol(var) );
      // G Feature 19:

      col_vals = SCIPcolGetVals(col);
	   SCIP_ROW** myrows = SCIPcolGetRows(col);
	   int nlprows = SCIPcolGetNLPNonz(col);

      edge_sum = 0;
	   edge_min = SCIPinfinity(scip);
	   edge_max = -SCIPinfinity(scip);

	   bias_sum = 0;
	   bias_min = SCIPinfinity(scip);
	   bias_max = -SCIPinfinity(scip);

	   obj_cos_sim_sum = 0.;
	   obj_cos_sim_min = 1.0;
	   obj_cos_sim_max = -1.0;

	   is_tight_sum = 0.;
	   is_tight_min = 1.0;
	   is_tight_max = -1.0;

	   dual_solution_sum = 0.;
	   dual_solution_min = SCIPinfinity(scip);
	   dual_solution_max = -SCIPinfinity(scip);

	   scaled_age_sum = 0.;
	   scaled_age_min = SCIPinfinity(scip);
	   scaled_age_max = -SCIPinfinity(scip);

      row_counter = 0;


	   for (int row_idx = 0; row_idx < nlprows; ++row_idx)
	   {
		   row = myrows[row_idx];
         row_lp_id = SCIProwGetLPPos(row);
		   row_norm = rows_reduced_norm[row_lp_id];
		
		   if ( !isnan( rows_lhs[row_lp_id] ) )  {
			
			   edge_val = -col_vals[row_idx] / row_norm; // in [-1,1]
			   edge_sum += edge_val;
			   edge_min = min(edge_min, edge_val);
			   edge_max = max(edge_max, edge_val);

			   bias_val = -1. * safe_div(rows_lhs[row_lp_id], row_norm);
			   bias_sum += bias_val;
			   bias_min = min(bias_min, bias_val);
			   bias_max = max(bias_max, bias_val);

			   obj_cos_sim_val = -1 * rows_reduced_obj_cos_sim[row_lp_id];
			   obj_cos_sim_sum += obj_cos_sim_val;
			   obj_cos_sim_min = min(obj_cos_sim_min, obj_cos_sim_val);
			   obj_cos_sim_max = max(obj_cos_sim_max, obj_cos_sim_val);

			   is_tight_val = is_at_lhs(scip, row);
			   is_tight_sum += is_tight_val;
			   is_tight_min = min(is_tight_min, is_tight_val);
			   is_tight_max = max(is_tight_max, is_tight_val);

			   dual_solution_val = -1. * safe_div(SCIProwGetDualsol(row), (row_norm * reduced_obj_norm) );
			   dual_solution_sum += dual_solution_val;
			   dual_solution_min = min(dual_solution_min, dual_solution_val);
			   dual_solution_max = max(dual_solution_max, dual_solution_val);

			   scaled_age_val = SCIProwGetAge(row) / (nlps + 5.0);
			   scaled_age_sum += scaled_age_val;
			   scaled_age_min = min(scaled_age_min, scaled_age_val);
			   scaled_age_max = max(scaled_age_max, scaled_age_val);

			   row_counter ++;
		   }

		   if ( !isnan( rows_rhs[row_lp_id] ) )  {
			
			   edge_val = col_vals[row_idx] / row_norm; // in [-1,1]
			   edge_sum += edge_val;
			   edge_min = min(edge_min, edge_val);
			   edge_max = max(edge_max, edge_val);

			   bias_val = safe_div(rows_rhs[row_lp_id], row_norm);
			   bias_sum += bias_val;
			   bias_min = min(bias_min, bias_val);
			   bias_max = max(bias_max, bias_val);

			   obj_cos_sim_val = rows_reduced_obj_cos_sim[row_lp_id];
			   obj_cos_sim_sum += obj_cos_sim_val;
			   obj_cos_sim_min = min(obj_cos_sim_min, obj_cos_sim_val);
			   obj_cos_sim_max = max(obj_cos_sim_max, obj_cos_sim_val);

			   is_tight_val = is_at_rhs(scip, row);
			   is_tight_sum += is_tight_val;
			   is_tight_min = min(is_tight_min, is_tight_val);
			   is_tight_max = max(is_tight_max, is_tight_val);

			   dual_solution_val = safe_div(SCIProwGetDualsol(row), (row_norm * reduced_obj_norm) );
			   dual_solution_sum += dual_solution_val;
			   dual_solution_min = min(dual_solution_min, dual_solution_val);
			   dual_solution_max = max(dual_solution_max, dual_solution_val);

			   scaled_age_val = SCIProwGetAge(row) / (nlps + 5.0);
			   scaled_age_sum += scaled_age_val;
			   scaled_age_min = min(scaled_age_min, scaled_age_val);
			   scaled_age_max = max(scaled_age_max, scaled_age_val);
			
			   row_counter ++;

		   }
	   }


      features[i * n_features + 92] = edge_sum / (double) row_counter; // G Feature 20: edge_mean
      features[i * n_features + 93] = edge_min; // G Feature 21: edge_min
      features[i * n_features + 94] = edge_max; // G Feature 22: edge_max

      features[i * n_features + 95] = bias_sum / (double) row_counter; // G Feature 23: bias_mean
      features[i * n_features + 96] = bias_min; // G Feature 24: bias_min
      features[i * n_features + 97] = bias_max; // G Feature 25: bias_max

      features[i * n_features + 98] = obj_cos_sim_sum / (double) row_counter; // G Feature 26: obj_cos_sim_mean
      features[i * n_features + 99] = obj_cos_sim_min; // G Feature 27: obj_cos_sim_min
      features[i * n_features + 100] = obj_cos_sim_max; // G Feature 28: obj_cos_sim_max

      features[i * n_features + 101] = is_tight_sum / (double) row_counter; // G Feature 29: is_tight_mean
      features[i * n_features + 102] = is_tight_min; // G Feature 30: is_tight_min
      features[i * n_features + 103] = is_tight_max; // G Feature 31: is_tight_max

      features[i * n_features + 104] = dual_solution_sum / (double) row_counter; // G Feature 32: dual_solution_mean
      features[i * n_features + 105] = dual_solution_min; // G Feature 33: dual_solution_min
      features[i * n_features + 106] = dual_solution_max; // G Feature 34: dual_solution_max

      features[i * n_features + 107] = scaled_age_sum / (double) row_counter; // G Feature 35: scaled_age_mean
      features[i * n_features + 108] = scaled_age_min; // G Feature 36: scaled_age_min
      features[i * n_features + 109] = scaled_age_max; // G Feature 37: scaled_age_max

   }

   // Done calculating features

   // Scale K features.

   for (j = 0; j < 72; j++) {
      double max_feat_val = -SCIPinfinity(scip);
      double min_feat_val = SCIPinfinity(scip);

      for(i = 0; i < nlpcands; i++) {

         var = lpcands[i];
         index = SCIPvarGetProbindex(var);
      
         max_feat_val = max(features[i * n_features + j], max_feat_val);
         min_feat_val = min(features[i * n_features + j], min_feat_val);

      }

      max_feat_val -= min_feat_val;
      for(i = 0; i < nlpcands; i++) {

         var = lpcands[i];
         index = SCIPvarGetProbindex(var);
      
         features[i * n_features + j] = qb_div(features[i * n_features + j] - min_feat_val, max_feat_val);
      }
   }

   // Calculate scores! We can skip subtracting the shift because it does not change the ranking of candidates
   
   double best_score = -SCIPinfinity(scip);
   int best_candidate = -1;


   for(j = 0; j < n_params; j++) {

      //Single term
      if( model_int[j * 5] == 1 )
      {
         // power = 1
         if (model_int[j * 5 + 2] == 1) {

            for(i = 0; i < nlpcands; i++)
               //scores[i] += ((features[i * n_features + model_int[j * 5 + 1]] - shift[j]) / scale[j] * coef[j]);
               scores[i] += ((features[i * n_features + model_int[j * 5 + 1]]) / scale[j] * coef[j]);
         }

         // power = 2
         else
         {
            for(i = 0; i < nlpcands; i++)
               //scores[i] += ((square(features[i * n_features + model_int[j * 5 + 1]]) - shift[j]) / scale[j] * coef[j]);
               scores[i] += ((square(features[i * n_features + model_int[j * 5 + 1]])) / scale[j] * coef[j]);

         }
      }

      //Two terms
      else {

         for(i = 0; i < nlpcands; i++)
            //scores[i] += ((features[i * n_features + model_int[j * 5 + 1]] * features[i * n_features + model_int[j * 5 + 3]] - shift[j]) / scale[j] * coef[j]);
            scores[i] += ((features[i * n_features + model_int[j * 5 + 1]] * features[i * n_features + model_int[j * 5 + 3]]) / scale[j] * coef[j]);
      }

   }


   for(i = 0; i < nlpcands; i++) {
      if (scores[i] > best_score) {
         best_score = scores[i];
         best_candidate = i;
      }
   }


   SCIP_CALL( SCIPbranchVar(scip, lpcands[best_candidate], NULL, NULL, NULL) );

   *result = SCIP_BRANCHED;

   return SCIP_OKAY;
}

SCIP_RETCODE SCIPincludeBranchruleSparse(
   SCIP*                 scip 
   )
{
   SCIP_BRANCHRULEDATA* branchruledata;
   SCIP_BRANCHRULE* branchrule;

   branchruledata = NULL;
   branchrule = NULL;
   /* include branching rule */
   SCIP_CALL( SCIPincludeBranchruleBasic(scip, &branchrule, BRANCHRULE_NAME, BRANCHRULE_DESC, BRANCHRULE_PRIORITY, BRANCHRULE_MAXDEPTH,
         BRANCHRULE_MAXBOUNDDIST, branchruledata) );
   assert(branchrule != NULL);

   SCIP_CALL( SCIPsetBranchruleExecLp(scip, branchrule, branchExeclpSparse) );



   return SCIP_OKAY;
}

/**@} */
