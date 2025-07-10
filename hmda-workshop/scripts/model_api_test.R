# ============================================================================
# HMDA LOAN APPROVAL MODEL - DOMINO API TEST SCRIPT
# ============================================================================
# This script tests the deployed model API with various scenarios
# Update the URL and API key with your actual values

library(httr)
library(jsonlite)
library(tidyverse)

# Configuration
MODEL_URL <- "https://se-demo.domino.tech:443/models/686ef93ed63d7454845b3a0a/latest/model"
API_KEY <- "g59Ee9FQwbqFPU37P38p5HXLLkCkrppBNlGAmJCpyvFcBtJhjdbBX5vu1ujBsciV"

# Test cases
test_cases <- list(
  # Test 1: High income, low DTI - likely approval
  list(
    name = "high_income_low_dti",
    description = "High income, low DTI ratio - expecting approval",
    data = list(
      loan_amount = 250000,
      income = 150000,
      debt_to_income = 15,
      loan_type = "1",
      loan_purpose = "1",
      occupancy_type = "1",
      age = 38
    )
  ),
  
  # Test 2: Low income, high DTI - likely denial
  list(
    name = "low_income_high_dti", 
    description = "Low income, high DTI ratio - expecting denial",
    data = list(
      loan_amount = 300000,
      income = 35000,
      debt_to_income = 65,
      loan_type = "1",
      loan_purpose = "1",
      occupancy_type = "1",
      age = 28
    )
  ),
  
  # Test 3: Medium income, medium DTI
  list(
    name = "medium_income_medium_dti",
    description = "Medium income, medium DTI ratio",
    data = list(
      loan_amount = 200000,
      income = 75000,
      debt_to_income = 33,
      loan_type = "1",
      loan_purpose = "1",
      occupancy_type = "1",
      age = 48
    )
  ),
  
  # Test 4: Young applicant with good metrics
  list(
    name = "young_good_metrics",
    description = "Young applicant with good financial metrics",
    data = list(
      loan_amount = 150000,
      income = 65000,
      debt_to_income = 22,
      loan_type = "1",
      loan_purpose = "1",
      occupancy_type = "1",
      age = 23
    )
  ),
  
  # Test 5: Senior applicant
  list(
    name = "senior_applicant",
    description = "Senior applicant with moderate loan",
    data = list(
      loan_amount = 180000,
      income = 70000,
      debt_to_income = 28,
      loan_type = "1",
      loan_purpose = "1",
      occupancy_type = "1",
      age = 72
    )
  )
)

# Function to call API
call_api <- function(input_data) {
  response <- POST(
    MODEL_URL,
    authenticate(API_KEY, API_KEY, type = "basic"),
    body = toJSON(list(data = input_data), auto_unbox = TRUE),
    content_type("application/json"),
    accept("application/json"),
    timeout(30)
  )
  
  return(response)
}

# Run tests
cat("========================================\n")
cat("HMDA Model API Test Results\n")
cat("========================================\n\n")

results <- list()

for (test in test_cases) {
  cat("Test:", test$name, "\n")
  cat("Description:", test$description, "\n")
  
  # Make API call
  start_time <- Sys.time()
  response <- call_api(test$data)
  end_time <- Sys.time()
  response_time <- as.numeric(end_time - start_time, units = "secs")
  
  if (status_code(response) == 200) {
    # Parse response
    parsed <- fromJSON(content(response, as = "text", encoding = "UTF-8"))
    
    # Extract result from Domino's wrapper
    result <- parsed$result
    
    cat("✓ Success\n")
    cat("  Decision:", result$decision, "\n")
    cat("  Probability:", round(result$probability, 3), "\n")
    cat("  Confidence:", round(result$confidence, 3), "\n")
    cat("  Loan-to-Income:", round(result$loan_to_income_ratio, 2), "\n")
    cat("  Response time:", round(response_time, 3), "seconds\n")
    
    # Store results
    results[[test$name]] <- data.frame(
      test_name = test$name,
      status = "SUCCESS",
      decision = result$decision,
      probability = result$probability,
      confidence = result$confidence,
      loan_to_income = result$loan_to_income_ratio,
      response_time = response_time,
      stringsAsFactors = FALSE
    )
    
  } else {
    cat("✗ Failed with status:", status_code(response), "\n")
    error_msg <- content(response, as = "text", encoding = "UTF-8")
    cat("  Error:", substr(error_msg, 1, 100), "\n")
    
    results[[test$name]] <- data.frame(
      test_name = test$name,
      status = "FAILED",
      decision = NA,
      probability = NA,
      confidence = NA,
      loan_to_income = NA,
      response_time = response_time,
      stringsAsFactors = FALSE
    )
  }
  
  cat("\n")
}

# Summarize results
cat("========================================\n")
cat("Summary\n")
cat("========================================\n")

results_df <- bind_rows(results)

# Success rate
success_rate <- sum(results_df$status == "SUCCESS") / nrow(results_df)
cat("Success rate:", round(success_rate * 100, 1), "%\n")

# Decision distribution
if (any(results_df$status == "SUCCESS")) {
  success_results <- results_df[results_df$status == "SUCCESS",]
  cat("\nDecision distribution:\n")
  decision_table <- table(success_results$decision)
  for (d in names(decision_table)) {
    cat("  ", d, ":", decision_table[d], "\n")
  }
  
  cat("\nAverage response time:", round(mean(success_results$response_time), 3), "seconds\n")
}

# Save results
write.csv(results_df, "/mnt/results/api_test_results_corrected.csv", row.names = FALSE)
cat("\nResults saved to: /mnt/results/api_test_results_corrected.csv\n")

# Create a simple plot if we have successful results
if (any(results_df$status == "SUCCESS")) {
  library(ggplot2)
  
  p <- ggplot(success_results, aes(x = test_name, y = probability, fill = decision)) +
    geom_bar(stat = "identity") +
    scale_fill_manual(values = c("Approved" = "green", "Denied" = "red")) +
    coord_flip() +
    labs(title = "Model Predictions by Test Case",
         x = "Test Case",
         y = "Probability") +
    theme_minimal()
  
  ggsave("/mnt/results/api_test_predictions.png", p, width = 8, height = 6)
  cat("Plot saved to: /mnt/results/api_test_predictions.png\n")
}