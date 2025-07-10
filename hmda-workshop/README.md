# HMDA Loan Approval Prediction Model

This project builds and deploys machine learning models to predict loan approval decisions using Home Mortgage Disclosure Act (HMDA) data.

## Overview

The project uses 2024 HMDA Modified LAR data from [FFIEC](https://ffiec.cfpb.gov/data-publication/modified-lar/2024) to train binary classification models that predict whether a loan application will be approved or denied.

## Workflow

1. **Model Training** (`hmda_model_training.R`)
   - Downloads HMDA 2024 data automatically
   - Performs feature engineering and data preprocessing
   - Trains Logistic Regression and Random Forest models
   - Uses MLflow for experiment tracking
   - Selects best model based on F1 score
   - Saves model artifacts for deployment

2. **Model Deployment**
   - Deploys best model to Domino Model API
   - Scoring function: `score_loan`
   - Handles all data transformations internally
   - Returns approval probability and decision

3. **Dashboard** (`hmda_dashboard.R`)
   - Interactive Shiny application for model insights
   - Features: Executive dashboard, loan predictor, data explorer, performance metrics
   - Launch with: `bash app.sh`

## Model API Test

Run `hmda_api_test.R` to test the deployed model:

```r
# Update these values with your deployment details:
MODEL_URL <- "https://your-domino-url/models/your-model-id/latest/model"
API_KEY <- "your-api-key"
```

The test script includes 5 scenarios covering different applicant profiles and validates model responses.

## Dashboard Files

- `hmda_dashboard.R` - Main Shiny application
- `app.sh` - Launcher script that installs dependencies and starts the dashboard
- Access at: `http://0.0.0.0:8888` when running

## Quick Start

1. **Train Model**: Run `hmda_model_training.R`
2. **Deploy Model**: Follow instructions in `DEPLOYMENT_INSTRUCTIONS.md`
3. **Test API**: Run `hmda_api_test.R` with your deployment URL
4. **Launch Dashboard**: Execute `bash app.sh`

## Input Features

- `loan_amount`: Loan amount in dollars
- `income`: Annual income in dollars  
- `debt_to_income`: DTI ratio (percentage)
- `loan_type`: Type of loan (1-4)
- `loan_purpose`: Purpose code
- `occupancy_type`: Occupancy code (1-3)
- `age`: Applicant age in years

## Contact

For questions, contact: marc.doan@dominodatalab.com

## Disclaimer

This code is provided as-is with no warranties or guarantees. The model is for demonstration purposes and should not be used as the sole basis for lending decisions. Ensure compliance with all applicable Fair Lending laws and regulations.