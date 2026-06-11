Customer Churn Prediction Assessment
Hi! This repository contains my complete solution for the churn prediction technical assessment.

1. Initial Observations (EDA)
When I first loaded up the data, the most obvious hurdle was the sparsity. Out of the 230 anonymized features, over 150 of them are missing between 95% and 100% of their data.

On top of that, we have a heavy class imbalance. Only about 7.3% of the dataset represents actual churners.

2. Preprocessing & Feature Engineering
My first instinct with missing data is usually to impute it, but with sparsity this extreme (and features being anonymized), doing mean or median imputation would just inject massive amounts of noise.

Instead, I treated the missingness itself as a behavioral signal. If this is platform data, a NaN likely just means the user didn't interact with that specific feature.

I created explicit binary flags (1 if missing, 0 if present) for the highly sparse columns.

I added a feature that counts the total missing values per row. This acts as a rough proxy for a user's overall "drop in activity."

For the categorical variables, I converted them to Pandas category types rather than one-hot encoding them, so my model could handle them natively without exploding the feature space.

3. Modeling Approach
I chose LightGBM for the pipeline. It handles sparse matrices beautifully, trains quickly, and natively digests categorical data.

To tackle the 7.3% imbalance, I avoided synthetic resampling (like SMOTE). Generating synthetic points in a highly sparse, anonymized vector space usually degrades performance. Instead, I used cost-sensitive learning by adjusting the scale_pos_weight parameter. This heavily penalizes the model during training if it misses a minority class instance (a churner).

4. Evaluation & Results
The problem statement noted that the business cost of missing a churner is much higher than the cost of a false alarm. Because of that, standard Accuracy and ROC-AUC aren't the best ways to measure success here.

I focused my evaluation on Recall (catching as many actual churners as possible) and the F2 Score (which mathematically weights Recall twice as heavily as Precision), alongside PR-AUC.

Final Test Set Metrics:

Recall (Churners): 0.23 (Captured 171 actual churners)

F2 Score: 0.2227

PR-AUC Score: 0.1402

Based on the model's Information Gain, the top predictors for churn are Var202, Var198, and Var199. In a real-world scenario, my next step would be to collaborate with the data engineering team to de-anonymize these specific features so we could build targeted retention campaigns around them.

5. What I'd do with more time
If this were heading to production and I had a few more weeks to iterate, I would:

Run a full Optuna hyperparameter tuning study to dial in the max_depth and learning rates.

Implement SHAP values to give the business side user-level explanations for why someone is flagged as high-risk.

Work with the stakeholders to determine the exact dollar cost of a false positive vs. a false negative, and use that ratio to manually tune the prediction threshold (e.g., lowering it from 0.5 to 0.3) to maximize ROI.
