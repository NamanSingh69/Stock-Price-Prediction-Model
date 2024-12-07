# Stock Return Prediction Project

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [File Structure](#file-structure)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Model Interpretation](#model-interpretation)
8. [Future Predictions](#future-predictions)
9. [Usage Instructions](#usage-instructions)
10. [Results](#results)
11. [Conclusion](#conclusion)
12. [Contact](#contact)

---

## Project Overview

The **Stock Return Prediction Project** aims to forecast the one-year percentage change in stock returns for companies across various sectors and market capitalizations. Leveraging a comprehensive dataset containing financial metrics and performance indicators, this project employs machine learning techniques, specifically the Random Forest Regressor, to build predictive models. The project not only focuses on prediction accuracy but also emphasizes model interpretability through feature importance analysis using SHAP values.

---

## Dataset Description

The dataset used in this project comprises detailed financial and performance metrics of companies, categorized based on market capitalization and market segments. Below is an overview of the dataset's structure:

### **Data Ordering and Structure**

- **Order:** The dataset is organized in **ascending order of company names**.
- **Total Records:** 2,972 observations (rows).
- **Total Features:** 48 columns.

### **Columns and Data Types**

| Column Name                                | Data Type   | Description |
|--------------------------------------------|-------------|-------------|
| **Name**                                   | Alphanumeric| Company name. |
| **Return over 1year**                      | Float       | Percentage change in stock return over one year. |
| **Market Capitalization**                  | Float       | Market cap in crore (10⁷ Indian rupees). |
| **Return on equity**                       | Float       | Current year's return on equity (%). |
| **Return on equity preceding year**        | Float       | Previous year's return on equity (%). |
| **Average return on equity 3Years**        | Float       | 3-year average return on equity (%). |
| **Average return on equity 5Years**        | Float       | 5-year average return on equity (%). |
| **Average return on equity 7Years**        | Float       | 7-year average return on equity (%). |
| **Average return on equity 10Years**       | Float       | 10-year average return on equity (%). |
| **Price to Earning**                       | Float       | Price-to-earning ratio. |
| **Historical PE 3Years**                   | Float       | 3-year historical PE ratio. |
| **Historical PE 5Years**                   | Float       | 5-year historical PE ratio. |
| **Historical PE 7Years**                   | Float       | 7-year historical PE ratio. |
| **Historical PE 10Years**                  | Float       | 10-year historical PE ratio. |
| **Sales growth 3Years**                    | Float       | 3-year sales growth (% change). |
| **Sales growth 5Years**                    | Float       | 5-year sales growth (% change). |
| **Sales growth 7Years**                    | Float       | 7-year sales growth (% change). |
| **Sales growth 10Years**                   | Float       | 10-year sales growth (% change). |
| **EPS growth 3Years**                      | Float       | 3-year EPS growth (% change). |
| **EPS growth 5Years**                      | Float       | 5-year EPS growth (% change). |
| **EPS growth 7Years**                      | Float       | 7-year EPS growth (% change). |
| **EPS growth 10Years**                     | Float       | 10-year EPS growth (% change). |
| **PEG Ratio**                              | Float       | Price/Earnings to Growth ratio. |
| **G Factor**                               | Integer     | G Factor score. |
| **Price to book value**                    | Float       | Price-to-book value ratio. |
| **Price to Sales**                         | Float       | Price-to-sales ratio. |
| **Piotroski score**                        | Integer     | Piotroski F-Score. |
| **NPM last year**                          | Float       | Net Profit Margin last year (%). |
| **NPM preceding year**                     | Float       | Net Profit Margin preceding year (%). |
| **Dividend yield**                         | Float       | Dividend yield (%). |
| **Dividend Payout Ratio**                  | Float       | Dividend payout ratio (%). |
| **Promoter holding**                       | Float       | Promoter holding (%). |
| **Public holding**                         | Float       | Public holding (%). |
| **Pledged percentage**                     | Float       | Pledged percentage (%). |
| **Altman Z Score**                         | Float       | Altman Z-Score. |
| **Quick ratio**                            | Float       | Quick ratio. |
| **Debt to equity**                         | Float       | Debt-to-equity ratio. |
| **Interest Coverage Ratio**                | Float       | Interest coverage ratio. |
| **PE Expanding**                           | Categorical | 'Y' or 'N' indicating if PE is expanding since the previous year. |
| **Book Value Expanding**                   | Categorical | 'Y' or 'N' indicating if book value is expanding since the previous year. |
| **Reducing Debt**                          | Categorical | 'Y' or 'N' indicating if the company is reducing debt since the previous year. |
| **Increasing Margin**                      | Categorical | 'Y' or 'N' indicating if the company's margin is increasing since the previous year. |
| **Valuation**                              | Categorical | 0 for bad, 1 for good, 'N' for insufficient data based on certain metrics. |
| **Business Model**                         | Categorical | 0 for bad, 1 for good, 'N' for insufficient data based on certain metrics. |
| **Growth**                                 | Categorical | 0 for bad, 1 for good, 'N' for insufficient data based on certain metrics. |
| **Financial Health**                       | Categorical | 0 for bad, 1 for good, 'N' for insufficient data based on certain metrics. |
| **Ownership**                              | Categorical | 0 for bad, 1 for good, 'N' for insufficient data based on certain metrics. |
| **Score**                                  | Integer     | Sum of the previous five categorical columns; 'N' is treated as 0. |

### **Market Capitalization Categories**

Companies are categorized based on their market capitalization in crore (10⁷ Indian rupees) as follows:

- **Micro:** 100 to 500 crore
- **Small:** 500 to 16,000 crore
- **Mid:** 16,000 to 50,000 crore
- **Large:** 50,000+ crore

### **Market Segments**

Companies are also categorized based on their market segments:

- **Auto**
- **Banks and Financials**
- **Consumer Goods**
- **IT**
- **Pharma**

---

## File Structure

The project repository is organized as follows:

```
├── README.md
├── data
│   ├── stockdata.csv
│   ├── All.csv
│   ├── Large+Mid+Small+Micro.csv
│   ├── Large+Mid+Small.csv
│   ├── Large.csv
│   ├── Mid.csv
│   ├── Small.csv
│   ├── Auto.csv
│   ├── Banks and Financials.csv
│   ├── Consumer Goods.csv
│   ├── IT.csv
│   └── Pharma.csv
├── marketcode.ipynb
├── marketcode.pdf
├── rf_feature_importance.png
├── shap_summary_bar.png
├── shap_summary_detail.png
├── prediction_summary_stats.csv
└── predicted_future_returns.csv
```

### **Description of Files and Folders**

- **`data/`**: Contains the raw and processed datasets.
  - **`stockdata.csv`**: The primary dataset containing all observations and features.
  - **Market Cap-Based CSVs:**
    - **`All.csv`**: Dataset encompassing all market capitalization categories.
    - **`Large+Mid+Small+Micro.csv`**: Includes all market caps.
    - **`Large+Mid+Small.csv`**: Excludes Micro-cap companies.
    - **`Large.csv`**, **`Mid.csv`**, **`Small.csv`**: Datasets filtered by individual market cap categories.
  - **Market Segment-Based CSVs:**
    - **`Auto.csv`**, **`Banks and Financials.csv`**, **`Consumer Goods.csv`**, **`IT.csv`**, **`Pharma.csv`**: Datasets filtered by respective market segments.

- **`marketcode.ipynb`**: Jupyter Notebook containing the complete machine learning pipeline, from data preprocessing to model training and evaluation.

- **`marketcode.pdf`**: PDF version of the Jupyter Notebook, including all outputs and visualizations.

- **`rf_feature_importance.png`**: Bar chart visualizing the feature importances as determined by the Random Forest model.

- **`shap_summary_bar.png`**: SHAP summary plot (bar type) showing the overall feature importance based on SHAP values.

- **`shap_summary_detail.png`**: Detailed SHAP summary plot, providing deeper insights into feature contributions.

- **`prediction_summary_stats.csv`**: CSV file containing summary statistics of the model's predictions, including mean, median, standard deviation, minimum, and maximum predicted returns.

- **`predicted_future_returns.csv`**: Detailed CSV file with predicted future returns for each company, including confidence intervals and top feature values influencing each prediction.

---

## Data Preprocessing

Data preprocessing is a critical step to ensure that the machine learning model receives clean, consistent, and well-formatted data. Here's a breakdown of the preprocessing steps undertaken:

### **1. Loading the Data**

```python
df = pd.read_csv('Data/stockdata.csv')
print(f"Initial data shape: {df.shape}")
```

- **Action:** Reads the primary dataset `stockdata.csv` into a pandas DataFrame `df`.
- **Output:** Displays the initial shape of the dataset, which is **(2972, 48)** indicating 2,972 observations and 48 features.

### **2. Preserving Original Data**

```python
current_data = df.copy()
```

- **Action:** Creates a deep copy of the original dataset to `current_data`. This preserved copy is crucial for generating future predictions without altering the original data during preprocessing.

### **3. Encoding Categorical Variables**

```python
categorical_columns = ['PE Expanding', 'Book Value Expanding', 'Reducing Debt', 
                      'Increasing Margin', 'Valuation', 'Bu Model', 
                      'Growth', 'Financial Health ', 'Ownership']

# Replace 'N' with 0 for binary columns
binary_columns = ['PE Expanding', 'Book Value Expanding', 'Reducing Debt', 'Increasing Margin']
for col in binary_columns:
    df[col] = df[col].map({'Y': 1, 'N': 0})
    current_data[col] = current_data[col].map({'Y': 1, 'N': 0})

# Replace 'N' with 0 for metric columns
metric_columns = ['Valuation', 'B Model', 'Growth', 'Financial Health ', 'Ownership']
for col in metric_columns:
    df[col] = pd.to_numeric(df[col].replace('N', '0'))
    current_data[col] = pd.to_numeric(current_data[col].replace('N', '0'))

# Store company names for later use
company_names = current_data['Name'].copy()
```

- **Action:**
  - **Binary Columns:** Converts 'Y' (Yes) to `1` and 'N' (No) to `0` for binary categorical variables.
  - **Metric Columns:** Replaces 'N' with `0` and ensures the columns are in numeric format.
  - **Preservation:** Extracts and stores company names for future reference.

- **Rationale:** Machine learning models require numerical input. Encoding categorical variables into numerical form allows the model to process and learn from these features effectively.

### **4. Identifying and Preparing Numerical Features**

```python
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns.remove('Return over 1year')  # Remove target variable
print(f"Number of features: {len(numeric_columns)}")

# Handle missing values
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
current_data[numeric_columns] = current_data[numeric_columns].fillna(df[numeric_columns].median())

# Fit scaler on all historical data
scaler = StandardScaler()
scaler.fit(df[numeric_columns])

# Transform both historical and current data
df[numeric_columns] = scaler.transform(df[numeric_columns])
current_data[numeric_columns] = scaler.transform(current_data[numeric_columns])

Number of features: 46
```

- **Action:**
  - **Feature Selection:** Identifies all numerical columns excluding the target variable `'Return over 1year'`, resulting in **46 features**.
  - **Handling Missing Values:** Imputes missing values in numerical features with the **median** of each respective column.
  - **Feature Scaling:** Applies `StandardScaler` to standardize features by removing the mean and scaling to unit variance.

- **Rationale:**
  - **Median Imputation:** The median is robust against outliers, ensuring that missing values are filled without being skewed by extreme values.
  - **Standardization:** Normalizing features ensures that each feature contributes equally to the model's learning process, preventing features with larger scales from dominating.

### **5. Target Variable Preparation**

```python
# Convert target variable
target_variable = 'Return over 1year'

# Remove extreme returns
df = df[df[target_variable] > -100]

# Convert to decimal
df['Return_decimal'] = df[target_variable] / 100

# Apply log transformation
df['Target'] = np.log1p(df['Return_decimal'])

# Remove any infinities or NaNs
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Target'])
print(f"Final shape after target preparation: {df.shape}")
```

- **Action:**
  - **Filtering Extreme Returns:** Removes any observations where the return is less than or equal to -100%, eliminating unrealistic or extreme outliers.
  - **Transformation:**
    - **Decimal Conversion:** Converts percentage returns to decimal form.
    - **Log Transformation:** Applies a natural logarithm transformation to stabilize variance and normalize the distribution of the target variable.
  - **Cleaning:** Eliminates any resulting infinite or NaN values after transformation.

- **Rationale:**
  - **Removing Extreme Values:** Prevents the model from being skewed by anomalous data points.
  - **Log Transformation:** Enhances the model's ability to capture multiplicative relationships and manage skewed data distributions.

- **Output:**
  
  ```
  Final shape after target preparation: (2688, 50)
  ```

  - **Meaning:** After preprocessing, the dataset has **2,688 observations** and **50 columns** (including the transformed target variables).

### **6. Feature Selection**

```python
# Cell 5: Feature Selection
X = df[numeric_columns]
y = df['Target']

# Select all features
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X, y)  # Fit on all historical data
selected_features = X.columns[selector.get_support()].tolist()

# Transform both historical and current data
X = pd.DataFrame(selector.transform(X), columns=selected_features, index=X.index)
X_current = pd.DataFrame(selector.transform(current_data[numeric_columns]), 
                        columns=selected_features, 
                        index=current_data.index)

print(f"Selected features shape: {X.shape}")

Selected features shape: (2688, 46)
```

- **Action:**
  - **Selection Criteria:** Uses `SelectKBest` with the `f_regression` scoring function to evaluate the linear relationship between each feature and the target variable.
  - **Number of Features Selected:** `k='all'` retains all **46 numerical features**.
  - **Transformation:** Applies the feature selection to both the training data (`X`) and the `current_data`, ensuring consistency.

- **Rationale:**
  - **SelectKBest:** Identifies features that have the strongest relationships with the target variable, potentially improving model performance by focusing on relevant predictors.
  - **Choosing `k='all'`:** In this instance, all features are retained, possibly because prior analysis indicated that each feature contributes meaningfully to the target prediction.

- **Output:**
  
  ```
  Selected features shape: (2688, 46)
  ```
  
  - **Meaning:** The dataset retains **46 features** after feature selection.

---

## Model Training and Evaluation

### **1. Splitting the Data**

```python
# Use TimeSeriesSplit to maintain temporal order
tscv = TimeSeriesSplit(n_splits=3) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

Training set shape: (2150, 46)
Testing set shape: (538, 46)
```

- **Action:**
  - **Cross-Validation Strategy:** Initializes `TimeSeriesSplit` with 3 splits to respect the temporal order of the data.
  - **Train-Test Split:** Divides the dataset into training (**2,150 observations**) and testing (**538 observations**) sets, allocating 80% to training and 20% to testing without shuffling.
  
- **Rationale:** Preserving the temporal sequence is crucial for time series data to prevent data leakage, ensuring that the model is trained on past data and tested on future data.

### **2. Hyperparameter Tuning with GridSearchCV**

```python
# Define parameter grid (adjusted for GridSearchCV)
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 7, 9, 11, 13],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'] 
}

# Create base model
rf = RandomForestRegressor(random_state=42)

# Create GridSearchCV object with updated param_grid
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    n_jobs=-1,
    verbose=2,
    scoring='neg_mean_squared_error'
)

# Fit on all historical data
print("Starting GridSearchCV with updated parameters")
grid_search.fit(X, y)
best_rf = grid_search.best_estimator_

# Print Grid Search results
print("\nGrid Search Results:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {-grid_search.best_score_:.4f} MSE")

# Get CV results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results.sort_values('rank_test_score')

# Print top 5 parameter combinations
print("\nTop 5 parameter combinations:")
print(cv_results[['params', 'mean_test_score', 'std_test_score']].head())

# Save the model
joblib.dump(best_rf, 'RandomForest_model.pkl')
print("\nModel saved as 'RandomForest_model.pkl'")
```

- **Action:**
  - **Parameter Grid Definition:** Specifies a comprehensive range of hyperparameters for the Random Forest model to explore.
    - **`n_estimators`:** Number of trees in the forest. Values from **100 to 500** test the trade-off between performance and computational efficiency.
    - **`max_depth`:** Maximum depth of each tree. Values from **5 to 13**, plus `None` (unlimited), assess the model's ability to capture complex patterns.
    - **`min_samples_split`:** Minimum number of samples required to split an internal node. Values from **2 to 8** help in controlling overfitting.
    - **`min_samples_leaf`:** Minimum number of samples required at a leaf node. Values **1, 2, 4** further regulate tree complexity.
    - **`max_features`:** Number of features to consider when looking for the best split. `'sqrt'` and `'log2'` introduce randomness and prevent overfitting.
  
  - **Model Initialization:** Sets up a `RandomForestRegressor` with a fixed `random_state` for reproducibility.
  
  - **GridSearchCV Setup:**
    - **Estimator:** The Random Forest model.
    - **Parameter Grid:** As defined above.
    - **Cross-Validation Strategy:** Utilizes `TimeSeriesSplit` to maintain temporal integrity.
    - **Parallel Processing:** `n_jobs=-1` leverages all available CPU cores to expedite the grid search.
    - **Verbose Level:** `verbose=2` provides detailed logs of the grid search progress.
    - **Scoring Metric:** `neg_mean_squared_error` (since GridSearchCV seeks to maximize the score).
  
  - **Execution:**
    - Initiates the grid search, which evaluates **720** parameter combinations across **5** folds (3 splits with TimeSeriesSplit typically produces multiple folds), totaling **3,600** model fits.
  
  - **Best Estimator Retrieval:** Extracts the model with the optimal hyperparameters based on cross-validation performance.
  
  - **Results Display:**
    - **Best Parameters:** Shows the hyperparameter combination that yielded the best performance.
    - **Best Score:** Displays the best negative MSE (converted back to positive for interpretability).
    - **Top 5 Parameter Combinations:** Lists the top-performing hyperparameter sets with their mean and standard deviation scores.
  
  - **Model Saving:** Persists the best model using `joblib` for future use, ensuring that the trained model can be deployed or analyzed without retraining.
  
- **Rationale:**
  - **Grid Search:** An exhaustive search over specified hyperparameters ensures that the model explores a wide range of configurations to identify the most effective combination.
  - **Cross-Validation Strategy:** `TimeSeriesSplit` respects the temporal nature of the data, ensuring that the model's evaluation reflects its performance on future, unseen data.

- **Output Interpretation:**
  
  ```
  Starting GridSearchCV with updated parameters
  Fitting 5 folds for each of 720 candidates, totalling 3600 fits
  
  Grid Search Results:
  Best parameters: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 6, 'n_estimators': 300}
  Best score: 0.2314 MSE
  
  Top 5 parameter combinations:
                                                 params  mean_test_score  \
  32  {'max_depth': None, 'max_features': 'sqrt', 'm...        -0.231415   
  34  {'max_depth': None, 'max_features': 'sqrt', 'm...        -0.231782   
  33  {'max_depth': None, 'max_features': 'sqrt', 'm...        -0.231809   
  22  {'max_depth': None, 'max_features': 'sqrt', 'm...        -0.231886   
  27  {'max_depth': None, 'max_features': 'sqrt', 'm...        -0.231886   
    
      std_test_score  
  32        0.048150  
  34        0.049256  
  33        0.048663  
  22        0.049182  
  27        0.049182  
    
  Model saved as 'RandomForest_model.pkl'
  ```
  
  - **Best Parameters:**
    - **`max_depth`:** `None` (no limit on tree depth).
    - **`max_features`:** `'sqrt'` (square root of total features).
    - **`min_samples_leaf`:** `2` (minimum of 2 samples per leaf node).
    - **`min_samples_split`:** `6` (minimum of 6 samples required to split an internal node).
    - **`n_estimators`:** `300` (number of trees in the forest).
  
  - **Best Score:** `0.2314 MSE` (Mean Squared Error), indicating the model's performance during cross-validation.
  
  - **Top 5 Parameter Combinations:** Show the most effective hyperparameter sets with their corresponding mean and standard deviation scores, highlighting the stability and performance consistency across different configurations.

### **3. Model Evaluation**

```python
y_pred = best_rf.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"MAPE: {mape:.2f}%")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

Model Performance:
MAPE: 78.01%
RMSE: 0.2152
R2 Score: 0.9009
```

- **Action:**
  - **Predictions:** Uses the best-trained model to predict returns on the testing set.
  - **Performance Metrics:**
    - **MAPE (Mean Absolute Percentage Error):** Measures the average absolute percentage difference between predicted and actual returns.
    - **RMSE (Root Mean Squared Error):** Measures the standard deviation of prediction errors.
    - **R² Score:** Indicates the proportion of variance in the target variable explained by the model.
  
- **Rationale:**
  - **MAPE:** Provides an intuitive percentage-based error metric.
  - **RMSE:** Sensitive to large errors, providing a measure of prediction accuracy in the same units as the target variable.
  - **R² Score:** Serves as a primary indicator of model performance, quantifying how well the model captures the underlying data patterns.
  
- **Output Interpretation:**
  
  ```
  Model Performance:
  MAPE: 78.01%
  RMSE: 0.2152
  R2 Score: 0.9009
  ```
  
  - **High R² Score (0.9009):** Indicates that **90.09%** of the variance in the target variable is explained by the model, suggesting a strong fit.
  - **High MAPE (78.01%):** Suggests that, on average, the model's predictions deviate from actual values by **78.01%**. This discrepancy warrants further investigation to understand the causes, such as data scaling issues or model limitations.
  - **RMSE (0.2152):** Represents the average magnitude of the prediction errors in the transformed target variable's scale.

---

## Model Interpretation

Understanding **why** a model makes certain predictions is as crucial as the predictions themselves. This project utilizes **SHAP (SHapley Additive exPlanations)** and traditional feature importance metrics to interpret the Random Forest model's decisions.

### **1. SHAP Values**

```python
# SHAP Values
explainer = shap.Explainer(best_rf)
shap_values = explainer(X_train)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_train, plot_type='bar', show=False)
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.savefig('shap_summary_bar.png', bbox_inches='tight', dpi=300)
plt.close()
```

- **Action:**
  - **Explainer Initialization:** Creates a SHAP explainer for the trained Random Forest model.
  - **SHAP Values Calculation:** Computes SHAP values for the training data, capturing each feature's contribution to individual predictions.
  - **Visualization:** Generates a bar plot summarizing feature importance based on SHAP values and saves it as `'shap_summary_bar.png'`.
  
- **Rationale:**
  - **SHAP Values:** Provide a unified measure of feature importance, offering insights into how each feature contributes to the model's predictions across all data points.
  
- **Output:**
  
  - **`shap_summary_bar.png`:** A bar chart displaying the overall feature importance based on SHAP values, highlighting the most influential features in the model.

### **2. Traditional Random Forest Feature Importance**

```python
# Random Forest Feature Importance
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': best_rf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.bar(range(len(feature_importance)), feature_importance['importance'])
plt.xticks(range(len(feature_importance)), feature_importance['feature'], rotation=45, ha='right')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('rf_feature_importance.png', bbox_inches='tight', dpi=300)
plt.close()
```

- **Action:**
  - **Feature Importance Calculation:** Extracts feature importance scores from the Random Forest model.
  - **Sorting:** Orders the features in descending order of importance.
  - **Visualization:** Creates a bar chart of feature importances and saves it as `'rf_feature_importance.png'`.
  
- **Rationale:**
  - **Random Forest Feature Importance:** Measures the average decrease in impurity (Gini impurity) brought by each feature across all trees in the forest, providing a global view of feature relevance.
  
- **Output:**
  
  - **`rf_feature_importance.png`:** A bar chart illustrating the traditional feature importance scores from the Random Forest model, allowing for comparison with SHAP-based importance.

### **3. Summary**

Both SHAP values and traditional feature importance provide valuable insights:

- **SHAP Values:**
  - Offer detailed, instance-level explanations.
  - Capture feature interactions and the directionality of feature impacts.
  
- **Random Forest Feature Importance:**
  - Provide a quick, global overview of feature relevance.
  - Are simpler to compute but may not capture complex interactions as effectively as SHAP.

---

## Future Predictions

Leveraging the trained model, the project generates future return predictions for each company, accompanied by confidence intervals to express prediction uncertainty.

### **1. Making Predictions**

```python
# Make predictions using current data
future_predictions = best_rf.predict(X_current)
```

- **Action:** Uses the trained Random Forest model to predict the transformed target variable (`'Target'`) for the entire dataset (`X_current`).

### **2. Inverse Transformation**

```python
# Create results DataFrame
results_df = pd.DataFrame({
    'Company_Name': company_names,
    'Predicted_Future_Return': np.expm1(future_predictions) * 100  # Convert back to percentage
})
```

- **Action:**
  - **Inverse Log Transformation:** Applies `np.expm1` (which computes `exp(x) - 1`) to revert the earlier log transformation, restoring the predictions to their original scale.
  - **Scaling Back to Percentage:** Multiplies by `100` to convert decimal returns back to percentage form.
  
- **Rationale:** Restoring the predictions to their original scale ensures interpretability and usability in real-world financial contexts.

### **3. Confidence Intervals Calculation**

```python
# Add prediction intervals using out-of-bag estimation
predictions = np.array([tree.predict(X_current) for tree in best_rf.estimators_])
prediction_std = np.std(predictions, axis=0)

# Calculate confidence intervals (95%)
results_df['Prediction_Std'] = prediction_std
results_df['Lower_Bound'] = np.expm1(future_predictions - 1.96 * prediction_std) * 100
results_df['Upper_Bound'] = np.expm1(future_predictions + 1.96 * prediction_std) * 100
```

- **Action:**
  - **Individual Tree Predictions:** Collects predictions from each tree in the Random Forest, resulting in an array of predictions.
  - **Standard Deviation Calculation:** Computes the standard deviation of these predictions for each company, serving as an estimate of prediction uncertainty.
  - **Confidence Intervals:**
    - **Lower Bound:** `expm1(prediction - 1.96 * std) * 100`
    - **Upper Bound:** `expm1(prediction + 1.96 * std) * 100`
  - **Interpretation:** These bounds represent the 95% confidence interval for each prediction, indicating where the true return is expected to lie with 95% certainty.
  
- **Rationale:** Confidence intervals provide a range of plausible values for each prediction, enhancing the reliability and interpretability of the forecasts.

### **4. Incorporating Top Feature Values**

```python
# Add feature importance for each prediction
top_features = feature_importance['feature'].head(5).tolist()
for feat in top_features:
    results_df[f'Feature_{feat}'] = current_data[feat]
```

- **Action:** For each company, the values of the top 5 most important features are added to the results DataFrame. This provides context on which features are influencing each prediction.
  
- **Rationale:** Understanding which features drive each prediction aids in making informed investment decisions and provides transparency into the model's workings.

### **5. Sorting and Saving Results**

```python
# Sort by predicted return
results_df = results_df.sort_values('Predicted_Future_Return', ascending=False)

# Save detailed results
results_df.to_csv('predicted_future_returns.csv', index=False)
print("\nFuture predictions saved to 'predicted_future_returns.csv'")
```

- **Action:**
  - **Sorting:** Orders the predictions in descending order, highlighting companies with the highest predicted returns.
  - **Saving:** Exports the detailed predictions to a CSV file for further analysis or reporting.
  
- **Rationale:** Organized and saved predictions facilitate easy access, comparison, and integration with other tools or dashboards.

### **6. Summary Statistics**

```python
# Create summary statistics
summary_stats = pd.DataFrame({
    'Metric': ['Mean Predicted Return', 'Median Predicted Return', 
               'Std Dev of Predictions', 'Min Predicted Return', 
               'Max Predicted Return'],
    'Value': [
        results_df['Predicted_Future_Return'].mean(),
        results_df['Predicted_Future_Return'].median(),
        results_df['Predicted_Future_Return'].std(),
        results_df['Predicted_Future_Return'].min(),
        results_df['Predicted_Future_Return'].max()
    ]
})

# Save summary statistics
summary_stats.to_csv('prediction_summary_stats.csv', index=False)
print("Summary statistics saved to 'prediction_summary_stats.csv'")
```

- **Action:**
  - **Metrics Computation:** Calculates key statistical measures of the predicted returns, including mean, median, standard deviation, minimum, and maximum.
  - **Saving:** Exports these summary statistics to a CSV file for quick reference.
  
- **Rationale:** Summary statistics provide a high-level overview of the prediction distribution, aiding in understanding the overall model performance and identifying any potential anomalies.

- **Output:**
  
  ```
  Future predictions saved to 'predicted_future_returns.csv'
  Summary statistics saved to 'prediction_summary_stats.csv'
  ```

  - **Meaning:** The pipeline successfully generated and saved detailed predictions and their summary statistics.

---

## Results

### **1. Model Performance Metrics**

```
Model Performance:
MAPE: 78.01%
RMSE: 0.2152
R2 Score: 0.9009
```

- **Interpretation:**
  - **R² Score (0.9009):** Indicates that the model explains **90.09%** of the variance in the target variable, suggesting a strong fit.
  - **MAPE (78.01%):** Reflects that, on average, the model's predictions deviate from actual values by **78.01%**. This high value suggests substantial prediction errors in percentage terms.
  - **RMSE (0.2152):** Measures the standard deviation of the prediction errors, providing an absolute measure of prediction accuracy.

### **2. Feature Importance Visualizations**

- **`rf_feature_importance.png`:** Displays the traditional feature importance scores from the Random Forest model, ranking features based on their contribution to reducing impurity across all trees.

- **`shap_summary_bar.png`:** Illustrates the feature importance based on SHAP values, offering a nuanced view of how each feature influences model predictions across all data points.

### **3. Future Predictions and Confidence Intervals**

- **`predicted_future_returns.csv`:** Contains predicted future returns for each company, including confidence intervals (`Lower_Bound` and `Upper_Bound`) and the values of the top 5 influential features for each prediction.

- **`prediction_summary_stats.csv`:** Provides summary statistics of the predicted returns, offering insights into the overall prediction distribution.

---

## Usage Instructions

### **Prerequisites**

Ensure that the following libraries are installed:

- Python 3.x
- pandas
- numpy
- scikit-learn
- shap
- matplotlib
- joblib
- scipy

You can install the required libraries using `pip`:

```bash
pip install pandas numpy scikit-learn shap matplotlib joblib scipy
```

### **Running the Project**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/NamanSingh69/Stock-Price-Prediction-Model.git
   cd stock-return-prediction
   ```

2. **Navigate to the Project Directory**

   Ensure you're in the root directory containing the `data/` folder and other project files.

3. **Run the Jupyter Notebook**

   Launch the Jupyter Notebook to execute the pipeline:

   ```bash
   jupyter notebook marketcode.ipynb
   ```

   Alternatively, you can view the project in the PDF format for a summarized version.

4. **Review Outputs**

   After running the notebook, the following outputs will be generated:

   - **Visualizations:**
     - `rf_feature_importance.png`
     - `shap_summary_bar.png`
     - `shap_summary_detail.png`

   - **Predictions:**
     - `predicted_future_returns.csv`
     - `prediction_summary_stats.csv`

   - **Model Persistence:**
     - `RandomForest_model.pkl` (trained model for future use)

### **Understanding the Outputs**

- **Visualizations:**
  - Analyze the feature importance charts to understand which financial metrics significantly influence the predicted returns.
  
- **Predictions:**
  - Use `predicted_future_returns.csv` to identify companies with the highest predicted returns, along with their confidence intervals and influential features.
  
- **Summary Statistics:**
  - Review `prediction_summary_stats.csv` to grasp the overall prediction distribution and variability.

---

## Conclusion

The **Stock Return Prediction Project** presents a comprehensive approach to forecasting one-year stock returns using financial metrics and performance indicators. By meticulously preprocessing data, selecting relevant features, and optimizing a Random Forest model through hyperparameter tuning, the project achieves robust predictive performance. Additionally, the incorporation of SHAP values and traditional feature importance metrics enhances the model's interpretability, providing actionable insights for stakeholders.

While the model demonstrates a strong R² score, the high MAPE underscores the need for further refinement, such as exploring alternative modeling techniques, enhancing feature engineering, or addressing potential data scaling issues. Continuous iteration and evaluation will drive improvements, ensuring that the model delivers reliable and precise predictions.

---

## Contact

For any queries, suggestions, or contributions, please contact:

- **Name:** [Naman Singh]
- **Email:** [namsingh419@example.com]
- **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/naman-singh419?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
- **GitHub:** [GitHub Profile](https://https://github.com/NamanSingh69/)

Feel free to reach out for collaboration opportunities or further discussions on stock return prediction and financial modeling.

---
