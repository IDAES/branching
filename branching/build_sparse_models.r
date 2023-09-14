library("glmnet")
library("pracma")
#install.packages("L0Learn", repos = "http://cran.rstudio.com")
library("L0Learn")

args = commandArgs(trailingOnly=TRUE)

print(args[1]) #seed
print(args[2]) #method

problem="setcover"

seed = args[1]
model=args[2]

center_colmeans <- function(x, xcenter) {
    x - rep(xcenter, rep.int(nrow(x), ncol(x)))
}

scale_apply <- function(x, x_sd) {
    x / rep(x_sd, rep.int(nrow(x), ncol(x)))
}

mode = 1
if (mode == 0) {
  print("Mode: simple. Original features will be used in the model.")
} else {
  print("Mode: quadratic. Products of features and squared terms can appear in the model.")
}

filename = paste("../data/", problem, "/datasets_preprocessed/", seed, "/train.csv", sep="")
train_df = read.csv(filename)

filename = paste("../data/", problem, "/datasets_preprocessed/", seed, "/valid.csv", sep="")
valid_df = read.csv(filename)

train_x = train_df[ ,! (colnames(train_df) == "Score")]
valid_x = valid_df[ ,! (colnames(valid_df) == "Score")]

train_y = train_df$Score
valid_y = valid_df$Score

if (mode == 0) {
  formula <- as.formula('Score ~ . -1')
} else {
  formula <- as.formula(paste('Score ~ .^2 + ',paste('poly(',colnames(train_x),',2, raw=TRUE)[, 2]',collapse = ' + '),paste('- 1')))
}

train_x$Score = train_y
valid_x$Score = valid_y

train_X = model.matrix(formula, train_x)
valid_X = model.matrix(formula, valid_x)

colmeans_train_X = colMeans(train_X)

train_X = center_colmeans(train_X, colmeans_train_X)
valid_X = center_colmeans(valid_X, colmeans_train_X)
print("Columns of X are centered")

train_X_sd = apply(train_X, 2, sd)
train_X_sd[train_X_sd == 0] = 1
print("Standard deviation for each column is calculated")

train_X = scale_apply(train_X, train_X_sd)
valid_X = scale_apply(valid_X, train_X_sd)
print("Columns of X are scaled to have a norm of 1")

ncols = length(colmeans_train_X)
df_mean_sd = cbind(1:ncols, colnames(train_X), colmeans_train_X, train_X_sd)

colnames(df_mean_sd) = c("Index", "Feature_name", "Mean", "Stddev")

model_dir = paste("../sparse-models/", problem, "/", seed, sep="")

dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

if (mode == 0) {
  write.csv(df_mean_sd, paste(model_dir, "/", model, "_means_sd_simple.csv", sep=""), row.names = FALSE)
} else {
  write.csv(df_mean_sd, paste(model_dir, "/", model, "_means_sd.csv", sep=""), row.names = FALSE)
}
mean_y = mean(train_y)

train_y <- scale(train_y, center = TRUE, scale = FALSE)
valid_y = valid_y - mean_y

print("Centering done for y")

if ( model == "glmnet_lasso" ) {
  my_model = glmnet(train_X, train_y, alpha = 1, intercept = FALSE, standardize = FALSE)
} else {
  penalty_type = strsplit(model,"_")[[1]][2]
  penalty_type = toupper(penalty_type)
  print("Penalty type: ")
  print(penalty_type)
  my_model = L0Learn.fit(train_X, train_y, penalty=penalty_type, nGamma = 5, gammaMin = 0.0001, gammaMax = 10, maxSuppSize=50, intercept = FALSE)
}

if (model == "glmnet_lasso") {
  feat_names = c("(Intercept)", colnames(train_X)) # glmnet adds a feature with this name
} else {
  feat_names = colnames(train_X) # l0learn does not add a feature with the name "(Intercept)"
}

if (mode == 0) {
  model_f_name = paste(model_dir, "/", model, "_model_verbose_simple.csv", sep="")
  logfile = paste(model_dir, "/", model, "_log_simple.txt", sep="")
} else {
  model_f_name = paste(model_dir, "/", model, "_model_verbose.csv", sep="")
  logfile = paste(model_dir, "/", model, "_log.txt", sep="")
}

cat(paste("Seed: ", seed, "\n\n", sep=""), file = logfile)
if (model == "glmnet_lasso") {

  grid = my_model$lambda

  n_lambda = length(grid)
  print("Grid length: ")
  print(n_lambda)

  preds = predict(my_model, train_X)
  valid_preds = predict(my_model, valid_X)

  for (i in 1:n_lambda) {

    my_train_pred = preds[,i]
    my_valid_pred = valid_preds[,i]
  
    train_mse = colMeans((my_train_pred - train_y)^2)
    valid_mse = mean((my_valid_pred - valid_y)^2)

    train_r2 = 1 - sum( (train_y - my_train_pred) ^2) / sum( (train_y - mean(train_y)) ^2)
    valid_r2 = 1 - sum( (valid_y - my_valid_pred) ^2) / sum( (valid_y - mean(valid_y)) ^2)

    if (i == 1) {
      best_train_r2 = train_r2
      best_valid_r2 = valid_r2
      best_train_mse = train_mse
      best_valid_mse = valid_mse

      best_model = 1
      model_coefs = coef(my_model)[,i]
      model_coefs_vec = unname(model_coefs)
      indices = which(model_coefs_vec != 0)
      if (length(indices) == 0) {
        write.csv(-1, model_f_name)
      }
      else {
        my_df = cbind(indices - 1, feat_names[indices], coef(my_model)[indices,i])
        colnames(my_df) = c("Index", "Feature_name", "Coefficient")
        write.csv(my_df, model_f_name, row.names = FALSE)
      }
    } else {
      
      if (valid_r2 > best_valid_r2) {
        best_train_r2 = train_r2
        best_valid_r2 = valid_r2
        best_train_mse = train_mse
        best_valid_mse = valid_mse
        best_model = i
        model_coefs = coef(my_model)[,i]
        model_coefs_vec = unname(model_coefs)
        indices = which(model_coefs_vec != 0)
        if (length(indices)== 0) {
          write.csv(-1, model_f_name)
        }
        else {
          my_df = cbind(indices - 1, feat_names[indices], coef(my_model)[indices,i])
          colnames(my_df) = c("Index", "Feature_name", "Coefficient")
          write.csv(my_df, model_f_name, row.names = FALSE)
        }

      }
    }
  
    line0 = paste(i, " Alpha = ", grid[i], " Nonzeros: ", length(indices), "\n", sep="")
    line1 = paste("Train MSE: ", train_mse, " Train R2: ", round(train_r2, 4), "\n", sep="")
    line2 = paste("Valid MSE: ", valid_mse, " Valid R2: ", round(valid_r2, 4), "\n\n", sep="")

    cat(line0, file = logfile, append = TRUE)
    cat(line1, file = logfile, append = TRUE)
    cat(line2, file = logfile, append = TRUE) 
  }
  line0 = paste("Best model: alpha = ", grid[best_model], " Train R2: ", best_train_r2, " Train MSE: ", best_train_mse, " Valid R2: ", best_valid_r2, " Valid MSE: ", best_valid_mse, "\n", sep="")
  cat(line0, file = logfile, append = TRUE)

  model_coefs = coef(my_model)[,best_model]
  model_coefs_vec = unname(model_coefs)
  indices = which(model_coefs_vec != 0)
} else {
  run_count = 1
  for (i in 1:5) {
    lambda_sequence = my_model$lambda[[i]]
    gamma_val = my_model$gamma[i]

    for (lam in lambda_sequence) {

      my_train_pred = predict(my_model, newx = train_X, lambda = lam, gamma = gamma_val)
      my_valid_pred = predict(my_model, newx = valid_X, lambda = lam, gamma = gamma_val)
    
      my_train_pred = as.data.frame(my_train_pred[,1])
      my_valid_pred = as.data.frame(my_valid_pred[,1])

      train_mse = colMeans((my_train_pred - train_y)^2)
      valid_mse = colMeans((my_valid_pred - valid_y)^2)

      train_r2 = 1 - sum( (train_y - my_train_pred) ^2) / sum( (train_y - mean(train_y)) ^2)
      valid_r2 = 1 - sum( (valid_y - my_valid_pred) ^2) / sum( (valid_y - mean(valid_y)) ^2)

      if (run_count == 1) {
        best_train_r2 = train_r2
        best_valid_r2 = valid_r2
        best_train_mse = train_mse
        best_valid_mse = valid_mse
        best_lambda = lam
        best_gamma = gamma_val

        best_model = 1
        model_coefs = coef(my_model,lambda = lam, gamma = gamma_val)
      
        indices = Matrix::which(model_coefs != 0)

        if (length(indices) == 0) {
          write.csv(-1, model_f_name)
        } else {
          my_df = cbind(indices, feat_names[indices], model_coefs[indices])
          colnames(my_df) = c("Index", "Feature_name", "Coefficient")
          write.csv(my_df, model_f_name, row.names = FALSE)

        }
      } else {
        if (valid_r2 > best_valid_r2) {
          best_train_r2 = train_r2
          best_valid_r2 = valid_r2
          best_train_mse = train_mse
          best_valid_mse = valid_mse
          best_lambda = lam
          best_gamma = gamma_val

          best_model = run_count
          model_coefs = coef(my_model,lambda = lam, gamma = gamma_val)
          indices = Matrix::which(model_coefs != 0)

          if (length(indices) == 0) {
            write.csv(-1, model_f_name)
          }
          else {
            my_df = cbind(indices, feat_names[indices], model_coefs[indices])
            colnames(my_df) = c("Index", "Feature_name", "Coefficient")
            write.csv(my_df, model_f_name, row.names = FALSE)
          }
        }
      }

      line0 = paste(run_count, " Gamma = ", gamma_val, " Lambda = ", lam, " Nonzeros: ", length(indices), "\n", sep="")
      line1 = paste("Train MSE: ", train_mse, " Train R2: ", round(train_r2, 4), "\n", sep="")
      line2 = paste("Valid MSE: ", valid_mse, " Valid R2: ", round(valid_r2, 4), "\n\n", sep="")

      cat(line0, file = logfile, append = TRUE)
      cat(line1, file = logfile, append = TRUE)
      cat(line2, file = logfile, append = TRUE) 

      if ((length(indices) > 0) & (length(indices) <= 10)) {
        line3 = paste("Features: ", feat_names[indices], "\n")
        cat(line3, file = logfile, append = TRUE)
      
      }
      run_count = run_count + 1
    }
  }

  line0 = paste("Best model: Gamma = ", best_gamma, " Lambda = ", best_lambda, " Train R2: ", best_train_r2, " Train MSE: ", best_train_mse, " Valid R2: ", best_valid_r2, " Valid MSE: ", best_valid_mse, "\n", sep="")
  cat(line0, file = logfile, append = TRUE)

  model_coefs = coef(my_model,lambda = best_lambda, gamma = best_gamma)
      
  indices = Matrix::which(model_coefs != 0)

}

# Create model file that the c program (sparse) will read

logfile = paste(model_dir, "/", model, "_model.txt", sep="")

cat(length(indices), file = logfile)
cat("\n", file = logfile, append = TRUE)

ctr = 0

for (name in feat_names[indices]) {
  
  ctr = ctr + 1
  index = indices[ctr]
  comps = strsplit(name,":")[[1]]

  nterms = 1

  feat1_id = -1
  feat1_pow = 1

  feat2_id = -1
  feat2_pow = 1

  if ( length(comps) == 1 ) { #single feature
    
    feat1 = comps[1]

    if( startsWith(feat1, "poly") ) { # squared term
      feat1_pow = 2
    }

    subcomps_list = strsplit(feat1,"_")[[1]]
    feat1_id = strtoi(subcomps_list[2])

    if ( endsWith(subcomps_list[1], "N") ) {
      feat1_id = feat1_id - 1 + 72
    } else if ( endsWith(subcomps_list[1], "K") ) {
      feat1_id = feat1_id - 1
    }
  } else { #interaction feature

    nterms = 2

    feat1 = comps[1]
    feat2 = comps[2]

    subcomps_list_1 = strsplit(feat1,"_")[[1]]

    feat1_id = strtoi(subcomps_list_1[2])

    if ( startsWith(feat1, "N") ) {
      feat1_id = feat1_id - 1 + 72
    } else {
      feat1_id = feat1_id - 1
    }

    subcomps_list_2 = strsplit(feat2,"_")[[1]]

    feat2_id = strtoi(subcomps_list_2[2])

    if ( startsWith(feat2, "N") ) {
      feat2_id = feat2_id - 1 + 72
    } else {
      feat2_id = feat2_id - 1
    }
  }
  
  line = paste(nterms, feat1_id, feat1_pow, feat2_id, feat2_pow, model_coefs[index], colmeans_train_X[index], train_X_sd[index], sep=",")
  if (ctr != length(indices))
    line = paste(line,"\n",sep="")
  cat(line, file = logfile, append = TRUE)

}
