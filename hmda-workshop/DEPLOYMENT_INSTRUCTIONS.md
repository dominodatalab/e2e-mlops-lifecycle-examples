# Domino Model API Deployment Instructions

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
     curl -X POST https://your-domino-url/models/your-model-id/latest/model \
       -H "Content-Type: application/json" \
       -H "Authorization: Bearer YOUR_API_KEY" \
       -d '{"data": {"loan_amount": 200000, "income": 50000, "debt_to_income": 35, "age": 38, ...}}'
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
- Monitor model performance through Domino's Model Monitoring
- Track prediction distributions and drift
- Set up alerts for unusual patterns

