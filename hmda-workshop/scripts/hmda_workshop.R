# ============================================================================
# HMDA LOAN APPROVAL PREDICTION MODEL
# ============================================================================
#
# MODEL OVERVIEW:
# This script builds binary classification models to predict loan approval decisions
# using Home Mortgage Disclosure Act (HMDA) data. The target variable is whether
# a loan application was approved (1) or denied (3).
#
# ============================================================================

# CONFIGURATION
# ============================================================================
MISSING_VALUE_STRATEGY <- "impute"  # "remove" or "impute"
EXPERIMENT_NAME <- "HMDA_Loan_Approval_Models_Workshop"
RUN_HYPERPARAMETER_SEARCH <- TRUE  # Set to FALSE for quick single runs
FORCE_CLEANUP <- TRUE  # Set to TRUE to force cleanup of all MLflow runs at start

# Define paths - script runs from /mnt/scripts
DATA_PATH <- "/mnt/data"
MODELS_PATH <- "/mnt/models"
RESULTS_PATH <- "/mnt/results"
PLOTS_PATH <- "/mnt/results/plots"

# Suppress warnings
options(rlang_warn_deprecated = FALSE, warning.length = 2000)
Sys.setenv(PYTHONWARNINGS = "ignore::UserWarning")

# Load required libraries
# Function to check and install packages
ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(paste("Installing", pkg, "...\n"))
    install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
  }
}

# List of required packages
required_packages <- c("mlflow", "tidyverse", "caret", "randomForest", "glmnet", "jsonlite")

# Check and install missing packages
cat("Checking required packages...\n")
for (pkg in required_packages) {
  ensure_package(pkg)
}

# Load required libraries
suppressPackageStartupMessages({
  library(mlflow)
  library(tidyverse)
  library(caret)
  library(randomForest)
  library(glmnet)
  library(jsonlite)  # For JSON export
})

cat("All packages loaded successfully!\n\n")

# Helper functions
quiet_mlflow <- function(expr) suppressWarnings(expr)

# Enhanced metrics calculation with proper AUC
calculate_metrics <- function(actual, predicted, probabilities = NULL) {
  # Ensure inputs are numeric and same length
  actual <- as.numeric(actual)
  predicted <- as.numeric(predicted)
  
  # Create confusion matrix
  conf_matrix <- table(Predicted = factor(predicted, levels = c(0, 1)),
                       Actual = factor(actual, levels = c(0, 1)))
  
  # Convert to numeric to avoid integer overflow
  TP <- as.numeric(conf_matrix[2,2])
  TN <- as.numeric(conf_matrix[1,1])
  FP <- as.numeric(conf_matrix[2,1])
  FN <- as.numeric(conf_matrix[1,2])
  
  # Basic metrics with safety checks
  total <- sum(conf_matrix)
  accuracy <- ifelse(total > 0, (TP + TN) / total, 0)
  
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  recall <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), 0)
  
  f1 <- ifelse((precision + recall) > 0,
               2 * (precision * recall) / (precision + recall), 0)
  
  balanced_accuracy <- (recall + specificity) / 2
  
  # Matthews Correlation Coefficient
  mcc_num <- (TP * TN) - (FP * FN)
  mcc_den_parts <- c((TP + FP), (TP + FN), (TN + FP), (TN + FN))
  
  if (any(mcc_den_parts == 0)) {
    mcc <- 0
  } else {
    mcc_den <- sqrt(prod(mcc_den_parts))
    mcc <- ifelse(mcc_den == 0, 0, mcc_num / mcc_den)
  }
  
  metrics <- list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1,
    specificity = specificity,
    balanced_accuracy = balanced_accuracy,
    mcc = mcc,
    confusion_matrix = conf_matrix
  )
  
  # Proper AUC calculation using Mann-Whitney U statistic
  if (!is.null(probabilities) && length(unique(actual)) == 2) {
    pos_probs <- probabilities[actual == 1]
    neg_probs <- probabilities[actual == 0]
    
    if (length(pos_probs) > 0 && length(neg_probs) > 0) {
      # Calculate all pairwise comparisons
      auc <- 0
      for (pos in pos_probs) {
        auc <- auc + sum(pos > neg_probs) + 0.5 * sum(pos == neg_probs)
      }
      auc <- auc / (length(pos_probs) * length(neg_probs))
      metrics$auc <- auc
    } else {
      metrics$auc <- 0.5
    }
  }
  
  return(metrics)
}

# Function to create performance visualizations
create_performance_plots <- function(metrics, model_name, save_dir = "/mnt/results/plots") {
  plots <- list()
  
  # 1. Confusion Matrix Heatmap
  conf_df <- as.data.frame(metrics$confusion_matrix)
  conf_df$Predicted <- factor(conf_df$Predicted, levels = c(0, 1))
  conf_df$Actual <- factor(conf_df$Actual, levels = c(0, 1))
  
  p1 <- ggplot(conf_df, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), size = 6) +
    scale_fill_gradient(low = "white", high = "darkblue") +
    labs(title = paste(model_name, "- Confusion Matrix"),
         x = "Actual", y = "Predicted") +
    theme_minimal() +
    theme(legend.position = "none")
  
  plots$confusion_matrix <- p1
  
  # 2. ROC Curve approximation
  if (!is.null(metrics$auc)) {
    roc_df <- data.frame(
      FPR = c(0, 1 - metrics$specificity, 1),
      TPR = c(0, metrics$recall, 1)
    )
    
    p2 <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
      geom_line(color = "blue", size = 1.5) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
      annotate("text", x = 0.7, y = 0.3,
               label = paste("AUC =", round(metrics$auc, 3)),
               size = 6) +
      labs(title = paste(model_name, "- ROC Curve"),
           x = "False Positive Rate",
           y = "True Positive Rate") +
      theme_minimal() +
      coord_equal() +
      xlim(0, 1) + ylim(0, 1)
    
    plots$roc_curve <- p2
  }
  
  # 3. Metrics Summary Plot
  metrics_df <- data.frame(
    Metric = c("Accuracy", "Precision", "Recall", "F1", "Specificity", "MCC"),
    Value = c(metrics$accuracy, metrics$precision, metrics$recall,
              metrics$f1_score, metrics$specificity, metrics$mcc)
  )
  
  p3 <- ggplot(metrics_df, aes(x = reorder(Metric, Value), y = Value)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label = round(Value, 3)), hjust = -0.1) +
    coord_flip() +
    ylim(0, 1.1) +
    labs(title = paste(model_name, "- Performance Metrics"),
         x = "", y = "Score") +
    theme_minimal()
  
  plots$metrics_summary <- p3
  
  return(plots)
}

# Function to create correlation heatmap
create_correlation_plot <- function(cor_matrix) {
  cor_df <- as.data.frame(as.table(cor_matrix))
  names(cor_df) <- c("Var1", "Var2", "Correlation")
  
  p <- ggplot(cor_df, aes(x = Var1, y = Var2, fill = Correlation)) +
    geom_tile() +
    geom_text(aes(label = round(Correlation, 2)), size = 3) +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                         midpoint = 0, limit = c(-1, 1)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Feature Correlations")
  
  return(p)
}

# Function to save MLflow artifacts safely
log_artifact_safe <- function(path) {
  tryCatch({
    invisible(capture.output(quiet_mlflow(mlflow_log_artifact(path))))
  }, error = function(e) {
    cat("Warning: Could not log artifact", path, "\n")
  })
}

# ============================================================================
# SETUP AND DATA LOADING
# ============================================================================

cat("=== HMDA Loan Approval Prediction Model ===\n")
cat("Starting experiment setup...\n\n")
cat("Working directory:", getwd(), "\n")
cat("Creating directories in parent folder (/mnt)...\n")

# Force cleanup if requested
if (FORCE_CLEANUP) {
  cat("Cleaning up any existing MLflow runs...\n")
  for (i in 1:5) {
    tryCatch({
      quiet_mlflow(mlflow_end_run())
    }, error = function(e) {})
  }
}

# Set up MLflow experiment
mlflow_set_experiment(EXPERIMENT_NAME)

# Create necessary directories
cat("Creating directories...\n")
dir.create(DATA_PATH, showWarnings = FALSE, recursive = TRUE)
dir.create(MODELS_PATH, showWarnings = FALSE, recursive = TRUE)
dir.create(RESULTS_PATH, showWarnings = FALSE, recursive = TRUE)
dir.create(PLOTS_PATH, showWarnings = FALSE, recursive = TRUE)

# Verify directories were created
if (dir.exists(DATA_PATH)) {
  cat("Data directory created at:", normalizePath(DATA_PATH), "\n")
} else {
  stop("Could not create data directory")
}

# Download HMDA data if not present
cat("Loading HMDA 2024 data...\n")
data_file <- "/mnt/data/hmda_2024.txt"
if (!file.exists(data_file)) {
  data_url <- "https://ffiec.cfpb.gov/file/modifiedLar/year/2024/institution/7H6GLXDRUGQFU57RNE97/txt"
  cat("Downloading to:", normalizePath(DATA_PATH, mustWork = FALSE), "\n")
  tryCatch({
    download.file(data_url, data_file, method = "curl", mode = "wb", quiet = FALSE)
    cat("Data downloaded successfully\n")
  }, error = function(e) {
    cat("Download failed. Trying alternative method...\n")
    tryCatch({
      download.file(data_url, data_file, method = "wget", mode = "wb", quiet = FALSE)
      cat("Data downloaded successfully with wget\n")
    }, error = function(e2) {
      cat("Download failed. Please check internet connection.\n")
      stop(e2)
    })
  })
}

# Define column names for HMDA data
col_names <- c(
  "activity_year", "lei", "loan_type", "loan_purpose", "preapproval",
  "construction_method", "occupancy_type", "loan_amount", "action_taken",
  "state", "county", "census_tract",
  "ethnicity_applicant_1", "ethnicity_applicant_2", "ethnicity_applicant_3",
  "ethnicity_applicant_4", "ethnicity_applicant_5",
  "ethnicity_coapplicant_1", "ethnicity_coapplicant_2", "ethnicity_coapplicant_3",
  "ethnicity_coapplicant_4", "ethnicity_coapplicant_5",
  "ethnicity_observed_applicant", "ethnicity_observed_coapplicant",
  "race_applicant_1", "race_applicant_2", "race_applicant_3",
  "race_applicant_4", "race_applicant_5",
  "race_coapplicant_1", "race_coapplicant_2", "race_coapplicant_3",
  "race_coapplicant_4", "race_coapplicant_5",
  "race_observed_applicant", "race_observed_coapplicant",
  "sex_applicant", "sex_coapplicant",
  "sex_observed_applicant", "sex_observed_coapplicant",
  "age_applicant", "age_above_62_applicant",
  "age_coapplicant", "age_above_62_coapplicant",
  "income", "purchaser_type", "rate_spread", "hoepa_status",
  "lien_status", "credit_score_applicant", "credit_score_coapplicant",
  "denial_reason_1", "denial_reason_2", "denial_reason_3", "denial_reason_4",
  "total_loan_costs", "total_points_fees", "origination_charges",
  "discount_points", "lender_credits", "interest_rate",
  "prepayment_penalty_term", "debt_to_income_ratio", "combined_loan_to_value_ratio",
  "loan_term", "intro_rate_period", "balloon_payment",
  "interest_only_payment", "negative_amortization", "other_nonamortizing_features",
  "property_value", "manufactured_home_secured_property_type",
  "manufactured_home_land_property_interest", "total_units",
  "multifamily_affordable_units", "submission_of_application",
  "initially_payable_to_institution", "aus_1", "aus_2", "aus_3", "aus_4", "aus_5",
  "reverse_mortgage", "open_end_line_of_credit", "business_or_commercial_purpose"
)

# Load and parse HMDA data
cat("Reading data file from:", data_file, "\n")
hmda_data <- read_delim(
  data_file,
  delim = "|",
  col_names = col_names,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE
)

cat("Loaded", nrow(hmda_data), "records\n")

# ============================================================================
# DATA PREPARATION AND FEATURE ENGINEERING
# ============================================================================

cat("\nPreparing data and engineering features...\n")

# Convert key numeric fields
hmda_clean <- hmda_data %>%
  mutate(
    loan_amount = as.numeric(loan_amount),
    income = as.numeric(income),
    property_value = as.numeric(property_value),
    # Note: interest_rate might only be available for originated loans
    # This could be a source of data leakage
    interest_rate = as.numeric(interest_rate),
    loan_type = as.factor(loan_type),
    loan_purpose = as.factor(loan_purpose),
    action_taken = as.factor(action_taken),
    occupancy_type = as.factor(occupancy_type),
    lien_status = as.factor(lien_status)
  )

# Filter for approved (1) and denied (3) applications only
hmda_model <- hmda_clean %>%
  filter(action_taken %in% c("1", "3")) %>%
  mutate(loan_approved = ifelse(action_taken == "1", 1, 0))

cat("Filtered to", nrow(hmda_model), "approved/denied applications\n")
cat("Approval rate:", round(mean(hmda_model$loan_approved), 3), "\n")

# Feature engineering with careful consideration of data leakage
hmda_features <- hmda_model %>%
  mutate(
    # Safe features (available at application time)
    loan_to_income = ifelse(!is.na(income) & income > 0,
                            loan_amount / income, NA),
    
    # Convert DTI ranges to numeric (midpoint of range)
    dti_numeric = case_when(
      debt_to_income_ratio == "<20%" ~ 15,
      debt_to_income_ratio == "20%-<30%" ~ 25,
      debt_to_income_ratio == "30%-<36%" ~ 33,
      debt_to_income_ratio == "36%-<50%" ~ 43,
      debt_to_income_ratio == "50%-60%" ~ 55,
      debt_to_income_ratio == ">60%" ~ 65,
      TRUE ~ NA_real_  # Missing DTI
    ),
    
    # Age groups
    age_group = case_when(
      age_applicant == "<25" ~ "Young",
      age_applicant %in% c("25-34", "35-44") ~ "Middle",
      age_applicant %in% c("45-54", "55-64") ~ "Older",
      age_applicant %in% c("65-74", ">74") ~ "Senior",
      TRUE ~ "Unknown"
    ),
    
    # Loan size categories
    loan_category = case_when(
      loan_amount < 100000 ~ "Small",
      loan_amount < 300000 ~ "Medium",
      loan_amount < 500000 ~ "Large",
      TRUE ~ "Jumbo"
    )
  )

# Important: Remove interest_rate as causes data leakage
# Interest rate is typically only known for originated loans
# Using it would give the model information about the outcome
cat("\nWARNING: Removing interest_rate to prevent data leakage\n")
cat("Interest rate is typically only available for originated loans\n")

# Select features carefully to avoid data leakage
features_to_use <- c(
  "loan_amount", "income", "loan_to_income",
  "dti_numeric", "loan_type", "loan_purpose", "occupancy_type",
  "age_group", "loan_category", "loan_approved"
)

model_data <- hmda_features %>%
  select(all_of(features_to_use)) %>%
  filter(!is.na(loan_amount) & !is.na(income) & loan_amount > 0 & income > 0) %>%
  mutate_if(is.character, as.factor)

cat("Final dataset:", nrow(model_data), "records with", ncol(model_data)-1, "features\n")

# ============================================================================
# START MAIN MLFLOW EXPERIMENT RUN
# ============================================================================

cat("\nStarting MLflow experiment tracking...\n")

# Ensure clean state
tryCatch(quiet_mlflow(mlflow_end_run()), error = function(e) {})
Sys.sleep(0.5)

# Start main parent run
main_run <- quiet_mlflow(mlflow_start_run())

# Log parent run experiment-level details
quiet_mlflow({
  # Parameters - experiment configuration
  mlflow_log_param("missing_value_strategy", MISSING_VALUE_STRATEGY)
  mlflow_log_param("hyperparameter_search", RUN_HYPERPARAMETER_SEARCH)
  mlflow_log_param("experiment_date", Sys.Date())
  mlflow_log_param("train_test_split", 0.8)
  mlflow_log_param("random_seed", 42)
  
  # Metrics - data statistics
  mlflow_log_metric("total_raw_records", nrow(hmda_data))
  mlflow_log_metric("filtered_records", nrow(hmda_model))
  mlflow_log_metric("final_records", nrow(model_data))
  mlflow_log_metric("n_features", length(features_to_use) - 1)
  mlflow_log_metric("target_approval_rate", mean(hmda_model$loan_approved))
})

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

cat("\nPerforming exploratory data analysis...\n")

# 1. Target distribution
p_target <- ggplot(hmda_model, aes(x = factor(loan_approved))) +
  geom_bar(fill = c("coral", "steelblue")) +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  scale_x_discrete(labels = c("Denied", "Approved")) +
  labs(title = "Loan Application Outcomes",
       subtitle = paste("Overall Approval Rate:",
                        round(mean(hmda_model$loan_approved) * 100, 1), "%"),
       x = "Decision", y = "Count") +
  theme_minimal()

ggsave("/mnt/results/plots/target_distribution.png", p_target, width = 8, height = 6)
log_artifact_safe("/mnt/results/plots/target_distribution.png")

# 2. Feature correlations
numeric_features <- model_data %>%
  select(where(is.numeric)) %>%
  select(-loan_approved)

if (ncol(numeric_features) > 1) {
  cor_matrix <- cor(numeric_features, use = "complete.obs")
  p_corr <- create_correlation_plot(cor_matrix)
  ggsave("/mnt/results/plots/feature_correlations.png", p_corr, width = 10, height = 8)
  log_artifact_safe("/mnt/results/plots/feature_correlations.png")
}

# 3. Missing value analysis
na_summary <- model_data %>%
  summarise_all(~sum(is.na(.)) / n() * 100) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing_pct") %>%
  filter(missing_pct > 0)

if (nrow(na_summary) > 0) {
  p_missing <- ggplot(na_summary, aes(x = reorder(variable, missing_pct), y = missing_pct)) +
    geom_bar(stat = "identity", fill = "darkred") +
    coord_flip() +
    labs(title = "Missing Values by Feature",
         x = "", y = "Missing %") +
    theme_minimal()
  
  ggsave("/mnt/results/plots/missing_values.png", p_missing, width = 8, height = 6)
  log_artifact_safe("/mnt/results/plots/missing_values.png")
}

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

cat("\nPreprocessing data...\n")

# Handle missing values based on strategy
preprocessing_params <- list(
  missing_value_strategy = MISSING_VALUE_STRATEGY,
  numeric_features = c("loan_amount", "income", "loan_to_income", "dti_numeric"),
  categorical_features = c("loan_type", "loan_purpose", "occupancy_type",
                           "age_group", "loan_category")
)

if (MISSING_VALUE_STRATEGY == "impute") {
  cat("Imputing missing values...\n")
  
  # Store imputation values for deployment
  imputation_values <- list()
  
  # Impute numeric features with median
  for (var in c("loan_to_income", "dti_numeric")) {
    if (var %in% names(model_data) && any(is.na(model_data[[var]]))) {
      median_val <- median(model_data[[var]], na.rm = TRUE)
      imputation_values[[var]] <- list(type = "median", value = median_val)
      model_data[[var]][is.na(model_data[[var]])] <- median_val
    }
  }
  
  preprocessing_params$imputation_values <- imputation_values
} else {
  cat("Removing records with missing values...\n")
  model_data <- model_data[complete.cases(model_data), ]
}

quiet_mlflow(mlflow_log_metric("n_samples_after_preprocessing", nrow(model_data)))

# Split data with stratification
set.seed(42)  # For reproducibility
train_index <- createDataPartition(model_data$loan_approved, p = 0.8, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# Ensure factor levels are aligned
factor_vars <- names(train_data)[sapply(train_data, is.factor)]
for (var in factor_vars) {
  all_levels <- unique(c(levels(train_data[[var]]), levels(test_data[[var]])))
  train_data[[var]] <- factor(train_data[[var]], levels = all_levels)
  test_data[[var]] <- factor(test_data[[var]], levels = all_levels)
}

# Save test data for Shiny app
saveRDS(test_data, "/mnt/models/test_data.rds")

# Log train/test statistics
quiet_mlflow({
  mlflow_log_metric("train_samples", nrow(train_data))
  mlflow_log_metric("test_samples", nrow(test_data))
  mlflow_log_metric("train_approval_rate", mean(train_data$loan_approved))
  mlflow_log_metric("test_approval_rate", mean(test_data$loan_approved))
})

cat("Training set:", nrow(train_data), "samples\n")
cat("Test set:", nrow(test_data), "samples\n")
cat("Training approval rate:", round(mean(train_data$loan_approved), 3), "\n")
cat("Test approval rate:", round(mean(test_data$loan_approved), 3), "\n")

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================================

cat("\n=== Training Logistic Regression Model ===\n")

# Start nested run for logistic regression
lr_run <- quiet_mlflow(mlflow_start_run(nested = TRUE))
quiet_mlflow({
  mlflow_set_tag("model_type", "logistic_regression")
})

# Prepare matrices for glmnet
x_train <- model.matrix(loan_approved ~ ., data = train_data)[, -1]
y_train <- train_data$loan_approved
x_test <- model.matrix(loan_approved ~ ., data = test_data)[, -1]
y_test <- test_data$loan_approved

# Hyperparameter search or single run
best_lr_model <- NULL
best_lr_metrics <- NULL
best_lr_alpha <- NULL
best_lr_f1 <- 0

if (RUN_HYPERPARAMETER_SEARCH) {
  cat("Performing hyperparameter search...\n")
  
  # Grid: 0=Ridge, 0.5=Elastic Net, 1=Lasso
  alpha_values <- c(0, 0.25, 0.5, 0.75, 1)
  
  for (alpha in alpha_values) {
    # Nested run for each alpha
    hp_run <- quiet_mlflow(mlflow_start_run(nested = TRUE))
    quiet_mlflow({
      mlflow_set_tag("model_type", "logistic_regression")
      
      # Log hyperparameters as parameters
      mlflow_log_param("alpha", alpha)
      mlflow_log_param("regularization_type",
                       ifelse(alpha == 0, "ridge",
                              ifelse(alpha == 1, "lasso", "elastic_net")))
    })
    
    # Train with cross-validation
    start_time <- Sys.time()
    set.seed(42)
    cv_model <- cv.glmnet(x_train, y_train, family = "binomial",
                          alpha = alpha, nfolds = 10)
    train_time <- as.numeric(Sys.time() - start_time, units = "secs")
    
    # Predictions
    pred_prob <- predict(cv_model, x_test, type = "response", s = "lambda.min")[,1]
    pred_class <- ifelse(pred_prob > 0.5, 1, 0)
    
    # Calculate metrics
    metrics <- calculate_metrics(y_test, pred_class, pred_prob)
    
    # Log metrics and additional parameters
    quiet_mlflow({
      mlflow_log_param("lambda_min", cv_model$lambda.min)
      mlflow_log_param("n_lambda", length(cv_model$lambda))
      
      mlflow_log_metric("train_time_seconds", train_time)
      mlflow_log_metric("accuracy", metrics$accuracy)
      mlflow_log_metric("precision", metrics$precision)
      mlflow_log_metric("recall", metrics$recall)
      mlflow_log_metric("f1_score", metrics$f1_score)
      mlflow_log_metric("specificity", metrics$specificity)
      mlflow_log_metric("mcc", metrics$mcc)
      if (!is.null(metrics$auc)) mlflow_log_metric("auc", metrics$auc)
    })
    
    # Track best model
    if (metrics$f1_score > best_lr_f1) {
      best_lr_f1 <- metrics$f1_score
      best_lr_model <- cv_model
      best_lr_metrics <- metrics
      best_lr_alpha <- alpha
    }
    
    quiet_mlflow(mlflow_end_run())
  }
  
  cat("Best alpha:", best_lr_alpha, "with F1:", round(best_lr_f1, 3), "\n")
} else {
  # Single run with default elastic net
  alpha <- 0.5
  quiet_mlflow({
    mlflow_log_param("alpha", alpha)
    mlflow_log_param("regularization_type", "elastic_net")
  })
  
  start_time <- Sys.time()
  set.seed(42)
  cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = alpha)
  train_time <- as.numeric(Sys.time() - start_time, units = "secs")
  
  pred_prob <- predict(cv_model, x_test, type = "response", s = "lambda.min")[,1]
  pred_class <- ifelse(pred_prob > 0.5, 1, 0)
  
  best_lr_model <- cv_model
  best_lr_metrics <- calculate_metrics(y_test, pred_class, pred_prob)
  best_lr_alpha <- alpha
  
  # Log parameters and metrics
  quiet_mlflow({
    mlflow_log_param("lambda_min", cv_model$lambda.min)
    mlflow_log_param("n_lambda", length(cv_model$lambda))
    
    mlflow_log_metric("train_time_seconds", train_time)
    mlflow_log_metric("accuracy", best_lr_metrics$accuracy)
    mlflow_log_metric("precision", best_lr_metrics$precision)
    mlflow_log_metric("recall", best_lr_metrics$recall)
    mlflow_log_metric("f1_score", best_lr_metrics$f1_score)
    mlflow_log_metric("specificity", best_lr_metrics$specificity)
    mlflow_log_metric("mcc", best_lr_metrics$mcc)
    if (!is.null(best_lr_metrics$auc)) mlflow_log_metric("auc", best_lr_metrics$auc)
  })
}

# Create visualizations
lr_plots <- create_performance_plots(best_lr_metrics, "Logistic Regression")
ggsave("/mnt/results/plots/lr_confusion_matrix.png", lr_plots$confusion_matrix, width = 6, height = 6)
ggsave("/mnt/results/plots/lr_metrics.png", lr_plots$metrics_summary, width = 8, height = 6)
if (!is.null(lr_plots$roc_curve)) {
  ggsave("/mnt/results/plots/lr_roc.png", lr_plots$roc_curve, width = 6, height = 6)
}

# Save model
saveRDS(best_lr_model, "/mnt/models/logistic_model.rds")
log_artifact_safe("/mnt/models/logistic_model.rds")
log_artifact_safe("/mnt/results/plots/lr_confusion_matrix.png")
log_artifact_safe("/mnt/results/plots/lr_metrics.png")

quiet_mlflow(mlflow_end_run())  # End LR run

# ============================================================================
# MODEL 2: RANDOM FOREST
# ============================================================================

cat("\n=== Training Random Forest Model ===\n")

# Start nested run for random forest
rf_run <- quiet_mlflow(mlflow_start_run(nested = TRUE))
quiet_mlflow({
  mlflow_set_tag("model_type", "random_forest")
})

best_rf_model <- NULL
best_rf_metrics <- NULL
best_rf_params <- NULL
best_rf_f1 <- 0

if (RUN_HYPERPARAMETER_SEARCH) {
  cat("Performing hyperparameter search...\n")
  
  # Reduced grid for efficiency
  rf_grid <- expand.grid(
    ntree = c(100, 200),
    mtry = c(2, 4, 6),
    nodesize = c(10, 20)
  )
  
  for (i in 1:nrow(rf_grid)) {
    params <- rf_grid[i, ]
    
    # Nested run for each combination
    hp_run <- quiet_mlflow(mlflow_start_run(nested = TRUE))
    quiet_mlflow({
      mlflow_set_tag("model_type", "random_forest")
      
      # Log hyperparameters as parameters
      mlflow_log_param("ntree", params$ntree)
      mlflow_log_param("mtry", params$mtry)
      mlflow_log_param("nodesize", params$nodesize)
    })
    
    # Train model
    start_time <- Sys.time()
    set.seed(42)
    rf_model <- randomForest(
      as.factor(loan_approved) ~ .,
      data = train_data,
      ntree = params$ntree,
      mtry = params$mtry,
      nodesize = params$nodesize,
      importance = TRUE
    )
    train_time <- as.numeric(Sys.time() - start_time, units = "secs")
    
    # Predictions
    pred_class <- as.numeric(as.character(predict(rf_model, test_data)))
    pred_prob <- predict(rf_model, test_data, type = "prob")[,2]
    
    # Calculate metrics
    metrics <- calculate_metrics(test_data$loan_approved, pred_class, pred_prob)
    
    # Log metrics
    quiet_mlflow({
      mlflow_log_metric("train_time_seconds", train_time)
      mlflow_log_metric("oob_error", rf_model$err.rate[nrow(rf_model$err.rate), 1])
      mlflow_log_metric("accuracy", metrics$accuracy)
      mlflow_log_metric("precision", metrics$precision)
      mlflow_log_metric("recall", metrics$recall)
      mlflow_log_metric("f1_score", metrics$f1_score)
      mlflow_log_metric("specificity", metrics$specificity)
      mlflow_log_metric("mcc", metrics$mcc)
      if (!is.null(metrics$auc)) mlflow_log_metric("auc", metrics$auc)
    })
    
    # Track best model
    if (metrics$f1_score > best_rf_f1) {
      best_rf_f1 <- metrics$f1_score
      best_rf_model <- rf_model
      best_rf_metrics <- metrics
      best_rf_params <- params
    }
    
    quiet_mlflow(mlflow_end_run())
  }
  
  cat("Best parameters - Trees:", best_rf_params$ntree,
      "Mtry:", best_rf_params$mtry,
      "F1:", round(best_rf_f1, 3), "\n")
} else {
  # Single run with default parameters
  params <- list(ntree = 100, mtry = 4, nodesize = 10)
  quiet_mlflow({
    mlflow_log_param("ntree", params$ntree)
    mlflow_log_param("mtry", params$mtry)
    mlflow_log_param("nodesize", params$nodesize)
  })
  
  start_time <- Sys.time()
  set.seed(42)
  rf_model <- randomForest(
    as.factor(loan_approved) ~ .,
    data = train_data,
    ntree = params$ntree,
    mtry = params$mtry,
    nodesize = params$nodesize,
    importance = TRUE
  )
  train_time <- as.numeric(Sys.time() - start_time, units = "secs")
  
  pred_class <- as.numeric(as.character(predict(rf_model, test_data)))
  pred_prob <- predict(rf_model, test_data, type = "prob")[,2]
  
  best_rf_model <- rf_model
  best_rf_metrics <- calculate_metrics(test_data$loan_approved, pred_class, pred_prob)
  best_rf_params <- params
  
  # Log metrics
  quiet_mlflow({
    mlflow_log_metric("train_time_seconds", train_time)
    mlflow_log_metric("oob_error", rf_model$err.rate[nrow(rf_model$err.rate), 1])
    mlflow_log_metric("accuracy", best_rf_metrics$accuracy)
    mlflow_log_metric("precision", best_rf_metrics$precision)
    mlflow_log_metric("recall", best_rf_metrics$recall)
    mlflow_log_metric("f1_score", best_rf_metrics$f1_score)
    mlflow_log_metric("specificity", best_rf_metrics$specificity)
    mlflow_log_metric("mcc", best_rf_metrics$mcc)
    if (!is.null(best_rf_metrics$auc)) mlflow_log_metric("auc", best_rf_metrics$auc)
  })
}

# Feature importance
importance_df <- importance(best_rf_model) %>%
  as.data.frame() %>%
  mutate(Feature = rownames(.)) %>%
  arrange(desc(MeanDecreaseGini)) %>%
  slice_head(n = 10)

# Save feature importance for Shiny app
saveRDS(importance_df, "/mnt/models/feature_importance.rds")

p_importance <- ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseGini),
                                          y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  labs(title = "Random Forest - Top 10 Feature Importance",
       x = "", y = "Mean Decrease in Gini") +
  theme_minimal()

ggsave("/mnt/results/plots/rf_importance.png", p_importance, width = 10, height = 8)

# Create visualizations
rf_plots <- create_performance_plots(best_rf_metrics, "Random Forest")
ggsave("/mnt/results/plots/rf_confusion_matrix.png", rf_plots$confusion_matrix, width = 6, height = 6)
ggsave("/mnt/results/plots/rf_metrics.png", rf_plots$metrics_summary, width = 8, height = 6)
if (!is.null(rf_plots$roc_curve)) {
  ggsave("/mnt/results/plots/rf_roc.png", rf_plots$roc_curve, width = 6, height = 6)
}

# Save model
saveRDS(best_rf_model, "/mnt/models/rf_model.rds")
log_artifact_safe("/mnt/models/rf_model.rds")
log_artifact_safe("/mnt/results/plots/rf_confusion_matrix.png")
log_artifact_safe("/mnt/results/plots/rf_metrics.png")
log_artifact_safe("/mnt/results/plots/rf_importance.png")

quiet_mlflow(mlflow_end_run())  # End RF run

# ============================================================================
# MODEL COMPARISON AND SELECTION
# ============================================================================

cat("\n=== Model Comparison ===\n")

# Create comparison table
model_comparison <- data.frame(
  Model = c("Logistic Regression", "Random Forest"),
  Accuracy = c(best_lr_metrics$accuracy, best_rf_metrics$accuracy),
  Precision = c(best_lr_metrics$precision, best_rf_metrics$precision),
  Recall = c(best_lr_metrics$recall, best_rf_metrics$recall),
  F1_Score = c(best_lr_metrics$f1_score, best_rf_metrics$f1_score),
  AUC = c(best_lr_metrics$auc, best_rf_metrics$auc),
  MCC = c(best_lr_metrics$mcc, best_rf_metrics$mcc)
)

print(model_comparison)

# Visualize comparison
model_comp_long <- model_comparison %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

p_comparison <- ggplot(model_comp_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("Logistic Regression" = "lightblue",
                               "Random Forest" = "darkgreen")) +
  labs(title = "Model Performance Comparison",
       subtitle = "Lower AUC values suggest less overfitting",
       y = "Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("/mnt/results/plots/model_comparison.png", p_comparison, width = 10, height = 6)
log_artifact_safe("/mnt/results/plots/model_comparison.png")

# Select best model based on F1 score
if (best_rf_metrics$f1_score > best_lr_metrics$f1_score) {
  best_model <- "Random Forest"
  best_model_obj <- best_rf_model
  best_metrics <- best_rf_metrics
} else {
  best_model <- "Logistic Regression"
  best_model_obj <- best_lr_model
  best_metrics <- best_lr_metrics
}

cat("\nBest model:", best_model, "\n")
cat("Performance metrics:\n")
cat("  F1 Score:", round(best_metrics$f1_score, 3), "\n")
cat("  AUC:", round(best_metrics$auc, 3), "\n")
cat("  Accuracy:", round(best_metrics$accuracy, 3), "\n")

# Check for overfitting
if (best_metrics$auc > 0.95) {
  cat("\nWARNING: High AUC (", round(best_metrics$auc, 3),
      ") suggests possible overfitting!\n")
  cat("Consider:\n")
  cat("- Reviewing features for data leakage\n")
  cat("- Adding regularization\n")
  cat("- Collecting more diverse training data\n")
  cat("- Using cross-validation for more robust evaluation\n")
}

# ============================================================================
# DEPLOYMENT PREPARATION
# ============================================================================

cat("\n=== Preparing Model for Domino Deployment ===\n")

# Save preprocessing parameters for deployment
preprocessing_config <- list(
  features = features_to_use[-length(features_to_use)],  # Exclude target
  numeric_features = preprocessing_params$numeric_features,
  categorical_features = preprocessing_params$categorical_features,
  imputation_values = preprocessing_params$imputation_values,
  model_type = best_model
)

# Save factor levels from training data for consistent scoring
factor_levels <- list()
for (var in preprocessing_params$categorical_features) {
  if (var %in% names(train_data)) {
    factor_levels[[var]] <- levels(train_data[[var]])
  }
}

# Save all model artifacts
saveRDS(best_model_obj, "/mnt/models/best_model.rds")
saveRDS(preprocessing_config, "/mnt/models/preprocessing_config.rds")
saveRDS(factor_levels, "/mnt/models/factor_levels.rds")

log_artifact_safe("/mnt/models/best_model.rds")
log_artifact_safe("/mnt/models/preprocessing_config.rds")
log_artifact_safe("/mnt/models/factor_levels.rds")

# Create Domino-compatible scoring API script
create_domino_scoring_api <- function() {
  scoring_code <- '# HMDA Loan Approval Model - Domino Model API
# This script provides a scoring endpoint for the trained model
# Accepts raw numeric values and handles all transformations internally

# Function to check and install packages
ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(paste("Installing", pkg, "...\n"))
    install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
  }
}

# List of required packages
required_packages <- c("dplyr", "randomForest", "glmnet", "jsonlite")

# Check and install missing packages
cat("Checking required packages...\n")
for (pkg in required_packages) {
  ensure_package(pkg)
}

# Load required libraries
suppressPackageStartupMessages({
  library(jsonlite)
  library(glmnet)
  library(randomForest)
  library(dplyr)
})

cat("All packages loaded successfully!\n\n")

# Define scoring function for Domino Model API
# Input format: {"data": {"loan_amount": 200000, "income": 50000, "debt_to_income": 35, "age": 38, ...}}
score_loan <- function(loan_amount, income, debt_to_income, loan_type,
                      loan_purpose, occupancy_type, age) {

  # Load model and configuration
  model <- readRDS("models/best_model.rds")
  config <- readRDS("models/preprocessing_config.rds")
  factor_levels <- readRDS("models/factor_levels.rds")

  # Validate inputs
  if (is.null(loan_amount) || is.na(loan_amount) || loan_amount <= 0) {
    return(list(error = "Invalid loan_amount", status = "error"))
  }

  if (is.null(income) || is.na(income) || income <= 0) {
    return(list(error = "Invalid income", status = "error"))
  }

  # Create input data frame
  input_data <- data.frame(
    loan_amount = as.numeric(loan_amount),
    income = as.numeric(income),
    loan_type = as.character(loan_type),
    loan_purpose = as.character(loan_purpose),
    occupancy_type = as.character(occupancy_type),
    stringsAsFactors = FALSE
  )

  # Apply feature engineering
  # 1. Loan-to-income ratio
  input_data$loan_to_income <- input_data$loan_amount / input_data$income

  # 2. Convert numeric DTI to dti_numeric (matching training data)
  if (!is.null(debt_to_income) && !is.na(debt_to_income)) {
    dti_value <- as.numeric(debt_to_income)
    input_data$dti_numeric <- case_when(
      dti_value < 20 ~ 15,
      dti_value >= 20 & dti_value < 30 ~ 25,
      dti_value >= 30 & dti_value < 36 ~ 33,
      dti_value >= 36 & dti_value < 50 ~ 43,
      dti_value >= 50 & dti_value <= 60 ~ 55,
      dti_value > 60 ~ 65,
      TRUE ~ 33  # Default
    )
  } else {
    # Use imputation value if available
    if (!is.null(config$imputation_values$dti_numeric)) {
      input_data$dti_numeric <- config$imputation_values$dti_numeric$value
    } else {
      input_data$dti_numeric <- 33  # Default
    }
  }

  # 3. Convert numeric age to age groups (matching training data)
  if (!is.null(age) && !is.na(age)) {
    age_value <- as.numeric(age)
    input_data$age_group <- case_when(
      age_value < 25 ~ "Young",
      age_value >= 25 & age_value < 35 ~ "Middle",
      age_value >= 35 & age_value < 45 ~ "Middle",
      age_value >= 45 & age_value < 55 ~ "Older",
      age_value >= 55 & age_value < 65 ~ "Older",
      age_value >= 65 & age_value < 75 ~ "Senior",
      age_value >= 75 ~ "Senior",
      TRUE ~ "Middle"  # Default
    )
  } else {
    input_data$age_group <- "Middle"  # Default
  }

  # 4. Create loan categories
  input_data$loan_category <- case_when(
    input_data$loan_amount < 100000 ~ "Small",
    input_data$loan_amount < 300000 ~ "Medium",
    input_data$loan_amount < 500000 ~ "Large",
    TRUE ~ "Jumbo"
  )

  # 5. Apply imputation for loan_to_income if needed
  if (!is.null(config$imputation_values$loan_to_income) && is.na(input_data$loan_to_income)) {
    input_data$loan_to_income <- config$imputation_values$loan_to_income$value
  }

  # 6. Convert categorical variables to factors
  for (var in config$categorical_features) {
    if (var %in% names(input_data)) {
      if (var %in% names(factor_levels)) {
        input_data[[var]] <- factor(input_data[[var]], levels = factor_levels[[var]])
      } else {
        input_data[[var]] <- as.factor(input_data[[var]])
      }
    }
  }

  # 7. Select model features
  model_features <- config$features

  # Check for missing features
  missing_features <- setdiff(model_features, names(input_data))
  if (length(missing_features) > 0) {
    return(list(
      error = paste("Missing required features:", paste(missing_features, collapse = ", ")),
      status = "error"
    ))
  }

  input_data <- input_data[, model_features, drop = FALSE]

  # Make predictions
  tryCatch({
    if (config$model_type == "Logistic Regression") {
      # Convert to matrix for glmnet
      input_matrix <- model.matrix(~ . - 1, data = input_data)
      pred_prob <- predict(model, input_matrix, type = "response", s = "lambda.min")
      if (is.matrix(pred_prob)) {
        pred_prob <- pred_prob[,1]
      }
    } else {
      # Random Forest
      pred_prob_matrix <- predict(model, input_data, type = "prob")
      if (is.matrix(pred_prob_matrix) || is.data.frame(pred_prob_matrix)) {
        pred_prob <- pred_prob_matrix[,2]
      } else {
        pred_prob <- pred_prob_matrix
      }
    }

    # Format response
    prediction <- ifelse(pred_prob > 0.5, 1, 0)

    result <- list(
      prediction = as.numeric(prediction),
      probability = round(as.numeric(pred_prob), 4),
      decision = ifelse(prediction == 1, "Approved", "Denied"),
      confidence = round(ifelse(prediction == 1, pred_prob, 1 - pred_prob), 4),
      model_type = config$model_type,
      status = "success",
      loan_to_income_ratio = round(input_data$loan_to_income, 2),

      # All features for drift monitoring
      features = list(
        # Raw inputs
        loan_amount = as.numeric(loan_amount),
        income = as.numeric(income),
        debt_to_income = as.numeric(debt_to_income),
        loan_type = as.character(loan_type),
        loan_purpose = as.character(loan_purpose),
        occupancy_type = as.character(occupancy_type),
        age = as.numeric(age),

        # Engineered features (important for monitoring)
        loan_to_income_ratio = round(input_data$loan_to_income, 2),
        dti_numeric = as.numeric(input_data$dti_numeric),
        age_group = as.character(input_data$age_group),
        loan_category = as.character(input_data$loan_category)
      ),

      # Metadata for monitoring
      model_metadata = list(
        model_version = "1.0",
        timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        prediction_id = paste0(
          "pred_", 
          format(Sys.time(), "%Y%m%d%H%M%S"), 
          "_", 
          sample(1000:9999, 1)
        )
      )
    )

    return(result)

  }, error = function(e) {
    return(list(
      error = paste("Prediction error:", e$message),
      status = "error"
    ))
  })
}

# Alternative function that accepts JSON input directly
score_json <- function(json_string) {
  # Parse JSON input
  input_list <- fromJSON(json_string)

  # Extract data (handle both {"data": {...}} and direct {...} formats)
  if ("data" %in% names(input_list)) {
    params <- input_list$data
  } else {
    params <- input_list
  }

  # Call scoring function
  result <- do.call(score_loan, params)

  # Return as JSON
  return(toJSON(result, auto_unbox = TRUE))
}

# Health check function
health_check <- function() {
  return(list(
    status = "healthy",
    model_loaded = file.exists("models/best_model.rds"),
    config_loaded = file.exists("models/preprocessing_config.rds"),
    levels_loaded = file.exists("models/factor_levels.rds"),
    timestamp = Sys.time()
  ))
}
'

writeLines(scoring_code, "/mnt/scoring_api.R")
}

# Create the scoring API
create_domino_scoring_api()
log_artifact_safe("/mnt/scoring_api.R")

# Create deployment instructions
deployment_instructions <- '# Domino Model API Deployment Instructions

## Prerequisites
1. Ensure your Domino environment has these R packages:
   - jsonlite
   - glmnet (if using Logistic Regression)
   - randomForest (if using Random Forest)
   - dplyr

## Deployment Steps

1. **Sync Model Files**

2. **Create Model API in Domino**
   - Go to Models > New Model
   - Select appropriate model environment
   - Set the prediction function to: `score_loan`
   - Deploy

3. **Configure Model Settings**
   - Model Name: HMDA Loan Approval Model
   - Description: Predicts loan approval (1) or denial (0) based on applicant data
   - Input Schema Example:
     ```json
     {
       "data": {
         "loan_amount": 200000,
         "income": 50000,
         "debt_to_income": 35,
         "loan_type": "1",
         "loan_purpose": "1",
         "occupancy_type": "1",
         "age": 38
       }
     }
     ```

4. **Test the Model**
   - Use the Domino Model API test interface
   - Or call via REST API:
     ```bash
     curl -X POST https://your-domino-url/models/your-model-id/latest/model \\
       -H "Content-Type: application/json" \\
       -H "Authorization: Bearer YOUR_API_KEY" \\
       -d \'{"data": {"loan_amount": 200000, "income": 50000, "debt_to_income": 35, "age": 38, ...}}\'
     ```

## Input Parameters

| Parameter | Type | Description | Example Values |
|-----------|------|-------------|----------------|
| loan_amount | numeric | Loan amount in dollars | 200000 |
| income | numeric | Annual income in dollars | 50000 |
| debt_to_income | numeric | Debt-to-income ratio (percentage) | 35 |
| loan_type | string | Type of loan | "1", "2", "3", "4" |
| loan_purpose | string | Purpose of loan | "1", "2", "31", "32", "4", "5" |
| occupancy_type | string | Occupancy type | "1", "2", "3" |
| age | numeric | Applicant age in years | 38 |

## Output Format

```json
{
  "prediction": 1,
  "probability": 0.8234,
  "decision": "Approved",
  "confidence": 0.8234,
  "model_type": "Random Forest",
  "status": "success",
  "loan_to_income_ratio": 2.67
}
```

## Monitoring
- Monitor model performance through Domino\'s Model Monitoring
- Track prediction distributions and drift
- Set up alerts for unusual patterns
'

writeLines(deployment_instructions, "/mnt/DEPLOYMENT_INSTRUCTIONS.md")
log_artifact_safe("/mnt/DEPLOYMENT_INSTRUCTIONS.md")

# ============================================================================
# MODEL CARD INFORMATION
# ============================================================================

# Create model card documentation
model_card <- list(
  model_details = list(
    name = "HMDA Loan Approval Model",
    version = "1.0",
    type = best_model,
    created_date = Sys.Date(),
    framework = ifelse(best_model == "Logistic Regression", "glmnet", "randomForest")
  ),
  
  intended_use = list(
    primary_use = "Predict loan approval decisions for HMDA applications",
    users = "Loan officers, risk analysts, compliance teams",
    out_of_scope = "Not for automated decision-making without human review"
  ),
  
  training_data = list(
    dataset = "HMDA 2024 Modified LAR",
    samples = nrow(train_data),
    features = length(features_to_use) - 1,
    target_distribution = list(
      approved = sum(train_data$loan_approved == 1),
      denied = sum(train_data$loan_approved == 0)
    )
  ),
  
  performance = list(
    test_set_size = nrow(test_data),
    metrics = list(
      f1_score = round(best_metrics$f1_score, 3),
      auc = round(best_metrics$auc, 3),
      accuracy = round(best_metrics$accuracy, 3),
      precision = round(best_metrics$precision, 3),
      recall = round(best_metrics$recall, 3)
    )
  ),
  
  limitations = list(
    "Model trained on 2024 data only",
    "Geographic coverage limited to reporting institutions",
    "Does not include all possible factors in lending decisions",
    ifelse(best_metrics$auc > 0.95,
           "High AUC suggests possible overfitting - use with caution",
           "Performance metrics within expected range")
  ),
  
  ethical_considerations = list(
    "Model should not be used as sole basis for lending decisions",
    "Regular monitoring for bias across protected classes required",
    "Compliance with Fair Lending laws mandatory"
  )
)

# Save model card
saveRDS(model_card, "/mnt/models/model_card.rds")
log_artifact_safe("/mnt/models/model_card.rds")

# Save as JSON
write(toJSON(model_card, pretty = TRUE), "/mnt/models/model_card.json")
log_artifact_safe("/mnt/models/model_card.json")

# End main MLflow run
quiet_mlflow(mlflow_end_run())

# ============================================================================
# FINAL SUMMARY
# ============================================================================

cat("\n=== Domino Deployment Summary ===\n")
cat("Model successfully prepared for Domino Model API deployment!\n")
cat("\nModel Details:\n")
cat("- Model Type:", best_model, "\n")
cat("- F1 Score:", round(best_metrics$f1_score, 3), "\n")
cat("- AUC:", round(best_metrics$auc, 3), "\n")
cat("- Accuracy:", round(best_metrics$accuracy, 3), "\n")

cat("\nFiles created for deployment:\n")
cat("- scoring_api.R: Main scoring function for Domino Model API\n")
cat("- models/best_model.rds: Trained model object\n")
cat("- models/preprocessing_config.rds: Preprocessing configuration\n")
cat("- models/factor_levels.rds: Factor levels for categorical variables\n")
cat("- DEPLOYMENT_INSTRUCTIONS.md: Step-by-step deployment guide\n")

cat("\nNext steps:\n")
cat("1. Sync all files to your Domino project\n")
cat("2. Create a new Model API in Domino\n")
cat("3. Set prediction function to: score_loan\n")
cat("4. Test with sample inputs\n")
cat("5. Monitor model performance\n")

cat("\nMLflow experiment tracking remains active for model comparison.\n")
cat("View experiments in MLflow UI to compare model runs and metrics.\n")

cat("\nScript completed successfully!\n")
