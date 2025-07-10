# HMDA Loan Approval Model - Domino Model API
# This script provides a scoring endpoint for the trained model
# Accepts raw numeric values and handles all transformations internally

# Function to check and install packages
ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(paste("Installing", pkg, "...
"))
    install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
  }
}

# List of required packages
required_packages <- c("dplyr", "randomForest", "glmnet", "jsonlite")

# Check and install missing packages
cat("Checking required packages...
")
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

cat("All packages loaded successfully!

")

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
      loan_to_income_ratio = round(input_data$loan_to_income, 2)
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

