#ifndef __SCIP_BRANCH_SPARSE_H__
#define __SCIP_BRANCH_SPARSE_H__


#include "scip/scip.h"
#include "scip/struct_lp.h"


extern double* coef;
extern double* shift;
extern double* scale;
extern int* model_int;

extern int n_params;

extern int n_features;

extern int n_static_features;

extern double* features;
extern double* static_features;

extern int n_K_static;
extern int n_G_static;

extern double* row_weights;
extern double* row_abs_sum_candidates;
extern double* row_abs_sum;
extern double* rows_positive_sum;
extern double* rows_negative_sum;
extern double* rows_rhs;
extern double* rows_lhs;
extern int* rows_nnz_nonfixed;

extern int row_abs_sum_size;

extern int* columns_fixed_or_not;
extern int* is_candidate;
extern double* scores;
extern double* rows_reduced_norm;
extern double* rows_reduced_obj_cos_sim;

extern int* rows_is_active;


/** creates the sparse branching rule and includes it in SCIP */
SCIP_RETCODE SCIPincludeBranchruleSparse(
   SCIP*                 scip                /**< SCIP data structure */
   );

#endif
