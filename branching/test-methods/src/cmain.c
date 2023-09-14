#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "scip/scip.h"
#include "scip/scipshell.h"
#include "scip/scipdefplugins.h"

#include "branch_sparse.h"

#define MAXCHAR 1000

double* coef;
double* shift;
double* scale;

int* model_int;

int n_params;

int n_K_static = 18;
int n_G_static = 5;

double* G_static;
double* G_dynamic;

int n_features = 110;
double* features;

int n_static_features = 23;
double* static_features;

double* row_weights;
double* row_abs_sum_candidates;
double* row_abs_sum;
double* rows_positive_sum;
double* rows_negative_sum;
double* rows_rhs;
double* rows_lhs;
int* rows_nnz_nonfixed;
int* columns_fixed_or_not;
int* is_candidate;
double* scores;
double* rows_reduced_norm;
double* rows_reduced_obj_cos_sim;

int row_abs_sum_size;
int* rows_is_active;

static
SCIP_RETCODE runShell(
   int                   argc,               /**< number of shell parameters */
   char**                argv               /**< array with shell parameters */
   )
{

   if (argc != 4) {
      printf("%d arguments are provided. Number of arguments should be 4.", argc);
      return -1;
   }

   SCIP* scip = NULL;
   SCIP_RETCODE retcode;
   char* problem_name;
   char* model_type;
   char status_str[50];
   int ml_model = 0;
   char branching_parameter[100];

   problem_name = argv[1];

   model_type = argv[2];
   printf("Model type: %s \n", model_type);

   if ((strcmp(model_type, "glmnet_lasso") == 0)|| (strcmp(model_type, "l0learn_l0l1") == 0)|| (strcmp(model_type, "l0learn_l0l2") == 0))
      ml_model = 1;
   else
      ml_model = 0;
   
   int main_seed = atoi(argv[3]);
   
   printf("Main seed: %d \n", main_seed);

   int seed1;
   int seed2;
   int seed3;
   int nvars, nconss;
   
   switch (main_seed) {
      case 0:
         seed1 = 1178568023;
         seed2 = 1273124120;
         seed3 = 1535857467;
         break;

      case 1:
         seed1 = 895547923;
         seed2 = 2141438070;
         seed3 = 1546885063;
         break;
      
      case 2:
         seed1 = 936291925;
         seed2 = 397460744;
         seed3 = 55676151;
         break;

      case 3:
         seed1 = 1182829494;
         seed2 = 151880525;
         seed3 = 1520735869;
         break;

      case 4:
         seed1 = 2076680766;
         seed2 = 1934069848;
         seed3 = 1175172316;
         break;

      default:
         printf("Seed not valid \n");
         return SCIP_ERROR;
   }


   SCIP_CALL( SCIPcreate(&scip) );

   SCIP_CALL( SCIPincludeDefaultPlugins(scip) );

   retcode = SCIPreadProb(scip, problem_name, NULL);

   switch( retcode )
   {
   case SCIP_NOFILE:
      SCIPinfoMessage(scip, NULL, "file <%s> not found\n", problem_name);
      return SCIP_OKAY;
   case SCIP_PLUGINNOTFOUND:
      SCIPinfoMessage(scip, NULL, "no reader for input file <%s> available\n", problem_name);
      return SCIP_OKAY;
   case SCIP_READERROR:
      SCIPinfoMessage(scip, NULL, "error reading file <%s>\n", problem_name);
      return SCIP_OKAY;
   default:
      SCIP_CALL( retcode );
   }
   
   SCIP_CALL( SCIPsetIntParam(scip, "presolving/maxrestarts", 0) );
   SCIP_CALL( SCIPsetIntParam(scip, "separating/maxrounds", 0) );
   SCIP_CALL( SCIPsetIntParam(scip, "timing/clocktype", 1) ); //1: CPU user time, 2: wallclock time
   SCIP_CALL( SCIPsetIntParam(scip, "lp/threads", 1) );
   SCIP_CALL( SCIPsetIntParam(scip, "parallel/maxnthreads", 1) );
   
   SCIP_CALL( SCIPsetIntParam(scip, "display/verblevel", 0) );
   SCIP_CALL( SCIPsetBoolParam(scip,"randomization/permuteconss", TRUE) );
   SCIP_CALL( SCIPsetBoolParam(scip,"randomization/permutevars", TRUE) );
	
   SCIP_CALL( SCIPsetIntParam(scip, "randomization/permutationseed", seed1) );
	SCIP_CALL( SCIPsetIntParam(scip, "randomization/randomseedshift", seed2) );
	SCIP_CALL( SCIPsetIntParam(scip, "randomization/lpseed", seed3) );
   
   SCIP_CALL( SCIPsetRealParam(scip, "limits/time", 3600) ); // 1 hour CPU time limit 
   if (ml_model != 1) {
      strcpy(branching_parameter, "branching/");
      strcat(branching_parameter, model_type);
      strcat(branching_parameter, "/");
      strcat(branching_parameter, "priority");

      SCIP_CALL( SCIPsetIntParam(scip, branching_parameter, 50000) );

   }
   
   if (ml_model) {

      SCIP_CALL( SCIPincludeBranchruleSparse(scip) );

      FILE* model_file;

      char model_file_name[100];

      strcpy(model_file_name, "../../../sparse-models/setcover/");
      strcat(model_file_name, argv[3]); //seed
      strcat(model_file_name, "/");
      strcat(model_file_name, model_type);
      strcat(model_file_name, "_model.txt");

      nvars = SCIPgetNVars(scip);
      nconss = SCIPgetNConss(scip);

      row_abs_sum_size = 2 * nconss;

      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &features, n_features * nvars) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &static_features, n_static_features * nvars) );

      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &row_weights, 4 * (row_abs_sum_size)) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &row_abs_sum_candidates, row_abs_sum_size) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &row_abs_sum, row_abs_sum_size) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &rows_positive_sum, row_abs_sum_size) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &rows_negative_sum, row_abs_sum_size) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &rows_rhs, row_abs_sum_size) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &rows_lhs, row_abs_sum_size) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &rows_nnz_nonfixed, row_abs_sum_size) );
      
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &columns_fixed_or_not, nvars) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &scores, nvars) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &is_candidate, nvars) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &rows_reduced_norm, row_abs_sum_size) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &rows_reduced_obj_cos_sim, row_abs_sum_size) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &rows_is_active, row_abs_sum_size) );

      char row[MAXCHAR];
      char* token;
      int ncoefficients = 0;
      int field_ct;

      //index,name,nterms,pow,coefficient,shift,scale
      model_file = fopen(model_file_name, "r");

      if (model_file == NULL) {
         printf("No model file found! \n");
         return SCIP_ERROR;
      }

      fgets(row, MAXCHAR, model_file);
      n_params = atoi(row);

      printf("Nparams: %d \n", n_params);

      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &coef, n_params) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &shift, n_params) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &scale, n_params) );

      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &model_int, 5 * n_params) );

      while (feof(model_file) != TRUE)
      {
         fgets(row, MAXCHAR, model_file);

         token = strtok(row, ",");
         
         field_ct = 0;
         while(token != NULL)
         {
            if (field_ct == 0) {
               model_int[ncoefficients * 5] = atoi(token); //nterms
            }
            
            else if (field_ct == 1) {
               model_int[ncoefficients * 5 + 1] = atoi(token); //term 1 id
            }
            else if (field_ct == 2) {
               model_int[ncoefficients * 5 + 2] = atoi(token); //term 1 power
            }
               
            else if (field_ct == 3) {
               model_int[ncoefficients * 5 + 3] = atoi(token); //term 2 id
            }
            else if (field_ct == 4) {
               model_int[ncoefficients * 5 + 4] = atoi(token); //term 2 power
            }
            else if (field_ct == 5) {
               double val;
               char* ptr;
               val = strtod(token, &ptr);
               coef[ncoefficients] = val;
            }
            else if (field_ct == 6) {
               double val;
               char* ptr;
               val = strtod(token, &ptr);
               shift[ncoefficients] = val;
            }
            else if (field_ct == 7) {
               double val;
               char* ptr;
               val = strtod(token, &ptr);
               scale[ncoefficients] = val;
            }

            field_ct ++;
            token = strtok(NULL, ",");
         }
         ncoefficients ++;
      }
      printf("Coefficients in the model: %d \n", ncoefficients);

      fclose(model_file);
   }

   struct timeval walltime_begin, walltime_end;
   struct tms cputime_begin, cputime_end;

   clock_t processtime_begin = clock();

   gettimeofday(&walltime_begin, NULL); //record walltime
   (void)times(&cputime_begin); //record user & system cpu time

   SCIP_CALL( SCIPsolve(scip) );

   clock_t processtime_end = clock();
   gettimeofday(&walltime_end, NULL);
   (void)times(&cputime_end);

   double processtime = (double)(processtime_end - processtime_begin) / CLOCKS_PER_SEC;
   double walltime = (double) ((walltime_end.tv_sec - walltime_begin.tv_sec) + (walltime_end.tv_usec - walltime_begin.tv_usec) * 1e-6);

   double cpu_user_time = (double) (cputime_end.tms_utime - cputime_begin.tms_utime) / sysconf(_SC_CLK_TCK);
   double cpu_system_time = (double) (cputime_end.tms_stime - cputime_begin.tms_stime) / sysconf(_SC_CLK_TCK);

   switch( SCIPgetStatus(scip) )
   {
   case SCIP_STATUS_UNKNOWN:
      strcpy(status_str, "unknown");
      break;
   case SCIP_STATUS_USERINTERRUPT:
      strcpy(status_str, "userinterrupt");
      break;
   case SCIP_STATUS_NODELIMIT:
      strcpy(status_str, "nodelimit");
      break;
   case SCIP_STATUS_TOTALNODELIMIT:
      strcpy(status_str, "totalnodelimit");
      break;
   case SCIP_STATUS_STALLNODELIMIT:
      strcpy(status_str, "stallnodelimit");
      break;
   case SCIP_STATUS_TIMELIMIT:
      strcpy(status_str, "timelimit");
      break;
   case SCIP_STATUS_MEMLIMIT:
      strcpy(status_str, "memlimit");
      break;
   case SCIP_STATUS_GAPLIMIT:
      strcpy(status_str, "gaplimit");
      break;
   case SCIP_STATUS_SOLLIMIT:
      strcpy(status_str, "sollimit");
      break;
   case SCIP_STATUS_BESTSOLLIMIT:
      strcpy(status_str, "bestsollimit");
      break;
   case SCIP_STATUS_RESTARTLIMIT:
      strcpy(status_str, "restartlimit");
      break;
   case SCIP_STATUS_OPTIMAL:
      strcpy(status_str, "optimal");
      break;
   case SCIP_STATUS_INFEASIBLE:
      strcpy(status_str, "infeasible");
      break;
   case SCIP_STATUS_UNBOUNDED:
      strcpy(status_str, "unbounded");
      break;
   case SCIP_STATUS_INFORUNBD:
      strcpy(status_str, "inforunbd");
      break;
   case SCIP_STATUS_TERMINATE:
      strcpy(status_str, "terminate");
      break;
   default:
      return SCIP_INVALIDDATA;
   }

   SCIP_BRANCHRULE* branchrule;
   
   if (ml_model)
      branchrule = SCIPfindBranchrule(scip, "sparse");
   else
      branchrule = SCIPfindBranchrule(scip, model_type);

   FILE* results;

   char results_file_name[200];

   strcpy(results_file_name, "../../../results/setcover/");
   strcat(results_file_name, model_type);
   strcat(results_file_name, "/");
   strcat(results_file_name, basename(problem_name));
   strcat(results_file_name, "_");
   strcat(results_file_name, argv[3]); //seed
   strcat(results_file_name, ".csv");

   results = fopen(results_file_name, "w");

   fprintf(results, "model,seed,instance,nnodes,nlps,nlpiterations,nnodelps,ndivinglps,nsbs,b_cutoffs,b_domreds,stime,gap,status,mem_used,mem_total,branching_time,walltime,proctime,cpu_user_time,cpu_system_time\n");
   fprintf(results, "%s,", model_type);
   fprintf(results, "%s,", argv[3]); //seed
   fprintf(results, "%s,", basename(problem_name) );
   fprintf(results, "%d,", (int) SCIPgetNNodes(scip));
   fprintf(results, "%d,", (int) SCIPgetNLPs(scip));
   fprintf(results, "%d,", (int) SCIPgetNLPIterations(scip));
   fprintf(results, "%d,", (int) SCIPgetNNodeLPs(scip));
   fprintf(results, "%d,", (int) SCIPgetNDivingLPs(scip));
   fprintf(results, "%d,", (int) SCIPgetNStrongbranchs(scip));
   fprintf(results, "%d,", (int) SCIPbranchruleGetNCutoffs(branchrule));
   fprintf(results, "%d,", (int) SCIPbranchruleGetNDomredsFound(branchrule));
   fprintf(results, "%f,", (double) SCIPgetSolvingTime(scip));
   fprintf(results, "%f,", (double) SCIPgetGap(scip));
   fprintf(results, "%s,", status_str);
   fprintf(results, "%f,", (double) SCIPgetMemUsed(scip) / 1048576.0 );
   fprintf(results, "%f,", (double) SCIPgetMemTotal(scip) / 1048576.0 );
   fprintf(results, "%f,", (double) SCIPbranchruleGetTime(branchrule));
   fprintf(results, "%f,", walltime);
   fprintf(results, "%f,", processtime);
   fprintf(results, "%f,", cpu_user_time);
   fprintf(results, "%f\n", cpu_system_time);

   fclose(results);

   if (ml_model) {

      SCIPfreeBlockMemoryArray(scip, &coef, n_params);
      SCIPfreeBlockMemoryArray(scip, &shift, n_params);
      SCIPfreeBlockMemoryArray(scip, &scale, n_params);

      SCIPfreeBlockMemoryArray(scip, &model_int, 5 * n_params);

      SCIPfreeBlockMemoryArray(scip, &features, n_features * nvars);

      SCIPfreeBlockMemoryArray(scip, &row_weights, 4 * (row_abs_sum_size));
      SCIPfreeBlockMemoryArray(scip, &row_abs_sum_candidates, row_abs_sum_size);
      SCIPfreeBlockMemoryArray(scip, &row_abs_sum, row_abs_sum_size);
      SCIPfreeBlockMemoryArray(scip, &rows_positive_sum, row_abs_sum_size);
      SCIPfreeBlockMemoryArray(scip, &rows_negative_sum, row_abs_sum_size); 
      SCIPfreeBlockMemoryArray(scip, &rows_rhs, row_abs_sum_size); 
      SCIPfreeBlockMemoryArray(scip, &rows_lhs, row_abs_sum_size); 
      SCIPfreeBlockMemoryArray(scip, &rows_nnz_nonfixed, row_abs_sum_size); 

      SCIPfreeBlockMemoryArray(scip, &columns_fixed_or_not, nvars);
      SCIPfreeBlockMemoryArray(scip, &is_candidate, nvars);
      SCIPfreeBlockMemoryArray(scip, &scores, nvars);
      SCIPfreeBlockMemoryArray(scip, &rows_reduced_norm, row_abs_sum_size);
      SCIPfreeBlockMemoryArray(scip, &rows_reduced_obj_cos_sim, row_abs_sum_size);
      SCIPfreeBlockMemoryArray(scip, &rows_is_active, row_abs_sum_size);
   
   }
   SCIP_CALL( SCIPfree(&scip) );

   BMScheckEmptyMemory();

   return SCIP_OKAY;
}

int main(
   int                        argc,
   char**                     argv
   )
{

   SCIP_RETCODE retcode;

   retcode = runShell(argc, argv);
   
   if( retcode != SCIP_OKAY )
   {
      SCIPprintError(retcode);
      return -1;
   }

   return 0;
}

