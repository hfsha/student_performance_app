# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("C:/Users/Shahidatul Hidayah/OneDrive/Documents/SEM 5/PRA/assgmt01_student_performance_dataset.csv")

# Step 1: Handle Missing Values
# Mean imputation for normally distributed columns
df["study_hours_per_week"].fillna(df["study_hours_per_week"].mean(), inplace=True)
df["attendance_rate"].fillna(df["attendance_rate"].mean(), inplace=True)

# Median imputation for skewed distributions
df["previous_exam_scores"].fillna(df["previous_exam_scores"].median(), inplace=True)
df["assignments_completed"].fillna(df["assignments_completed"].median(), inplace=True)

# Mode imputation for categorical-like column
df["extracurricular_participation"].fillna(df["extracurricular_participation"].mode()[0], inplace=True)

# Recalculate interaction term
df["study_attendance_interaction"].fillna(
    df["study_hours_per_week"] * df["attendance_rate"], inplace=True
)

# Impute missing values for squared terms using mean
df["study_hours_per_week_squared"].fillna(df["study_hours_per_week_squared"].mean(), inplace=True)
df["attendance_rate_squared"].fillna(df["attendance_rate_squared"].mean(), inplace=True)

# Verify that all missing values have been handled
print("Missing Values After Imputation:\n", df.isnull().sum())

# Step 2: Data Visualization
# Histogram of the target variable
plt.figure(figsize=(8, 5))
sns.histplot(df['final_exam_score'], kde=True)
plt.title('Distribution of Final Exam Scores')
plt.xlabel('Final Exam Score')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()

# 3D Scatter Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['study_hours_per_week'], df['attendance_rate'], df['final_exam_score'], c='b', marker='o')
ax.set_xlabel('Study Hours per Week')
ax.set_ylabel('Attendance Rate (%)')
ax.set_zlabel('Final Exam Score')
plt.title('3D Scatter Plot')
plt.show()

# Box Plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='extracurricular_participation', y='final_exam_score', data=df)
plt.title('Box Plot of Final Exam Scores by Extracurricular Participation')
plt.xlabel('Extracurricular Participation (0 = No, 1 = Yes)')
plt.ylabel('Final Exam Score')
plt.show()

# Step 3: Multicollinearity & Heteroscedasticity Evaluation
# Calculate Variance Inflation Factor (VIF)
features = df.drop(columns=['final_exam_score'])
vif_data = pd.DataFrame()
vif_data['Feature'] = features.columns
vif_data['VIF'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
print("Variance Inflation Factor (VIF):\n", vif_data)

# Check for heteroscedasticity using a residual plot
X = df[['study_hours_per_week', 'attendance_rate', 'previous_exam_scores', 'assignments_completed']]
y = df['final_exam_score']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
residuals = model.resid

plt.figure(figsize=(8, 5))
sns.scatterplot(x=model.fittedvalues, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Heteroscedasticity Check')
plt.show()

# Step 4: Model Development
X = df.drop(columns=['final_exam_score'])
y = df['final_exam_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_pred = linear_model.predict(X_test_scaled)

# Lasso Regression Model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)

# Ridge Regression Model
ridge = Ridge(alpha=0.1)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)

# Step 5: Model Evaluation
# Evaluate model performance
linear_mse = mean_squared_error(y_test, linear_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)

linear_r2 = r2_score(y_test, linear_pred)
lasso_r2 = r2_score(y_test, lasso_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print(f"Linear Regression MSE: {linear_mse:.4f}, R² Score: {linear_r2:.4f}")
print(f"Lasso Regression MSE: {lasso_mse:.4f}, R² Score: {lasso_r2:.4f}")
print(f"Ridge Regression MSE: {ridge_mse:.4f}, R² Score: {ridge_r2:.4f}")

# Feature Importance Analysis
important_features = X.columns[lasso.coef_ != 0]
print("Features Selected by Lasso Regression:", important_features)

# Step 6: Conclusion
print("\nConclusion:")
if lasso_r2 > ridge_r2 and lasso_r2 > linear_r2:
    print("Lasso Regression performed the best due to its feature selection capability.")
elif ridge_r2 > lasso_r2 and ridge_r2 > linear_r2:
    print("Ridge Regression performed the best due to handling multicollinearity effectively.")
else:
    print("Linear Regression performed the best, indicating regularization might not be necessary.")
