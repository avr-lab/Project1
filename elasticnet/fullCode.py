"""
Machine Learning 

Ajay Anand - A20581927
Anish Viswanathan - A20596106
Mohith Panchatcharam - A20562455
Sibi Chandra Sekar - A20577946
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
df = pd.read_csv(r'C:\Users\vrani\Desktop\machine_learning\ParisHousing.csv')
df.head()
df.shape
df.columns
df.isnull().sum()
df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

correlations = df.corr()['price'].sort_values(ascending=False)
print("Correlations with price:\n", correlations)

low_correlation_features = correlations[correlations.abs() < 0.1].index
df.drop(low_correlation_features, axis=1, inplace=True)

for col in df.columns:
    if col != 'price':
        upper_bound = df[col].quantile(0.99)
        lower_bound = df[col].quantile(0.01)
        df[col] = np.clip(df[col], lower_bound, upper_bound)
      numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = (df[col] - mean) / std
df.head()
X = df.drop('price', axis=1)
y = df['price']

X = sm.add_constant(X)

def elastic_net_loss(beta, X, y, alpha=0.001, l1_ratio=0.5):
    """
    Loss function for ElasticNet regression (L1 + L2 regularization)
    beta: coefficients (weights)
    X: feature matrix (with significant features)
    y: target variable
    alpha: regularization strength (controls the penalty)
    l1_ratio: balance between L1 and L2 (1.0 = L1, 0.0 = L2)
    """
    predictions = np.dot(X, beta)  # Predicted values
    residuals = y - predictions     # Error in predictions
    

    l1_penalty = np.sum(np.abs(beta))
    l2_penalty = np.sum(beta**2)
    
    # ElasticNet loss function: squared loss + L1 + L2
    loss = np.sum(residuals**2) / (2 * len(y)) + alpha * (l1_ratio * l1_penalty + (1 - l1_ratio) * l2_penalty)
    return loss

alphas = [0.0001, 0.001, 0.01, 0.1]  # Regularization strength values
l1_ratios = [0.1, 0.5, 0.9]  # Balance between L1 and L2
best_r2 = -np.inf
best_alpha = None
best_l1_ratio = None
best_beta = None

for alpha in alphas:
    for l1_ratio in l1_ratios:
        
        np.random.seed(42)
        initial_beta = np.random.normal(size=X.shape[1])

        result = minimize(elastic_net_loss, initial_beta, args=(X, y, alpha, l1_ratio), method='BFGS')

        optimal_beta = result.x

        predictions = np.dot(X, optimal_beta)

        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - predictions) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        if r_squared > best_r2:
            best_r2 = r_squared
            best_alpha = alpha
            best_l1_ratio = l1_ratio
            best_beta = optimal_beta

        print(f"Alpha: {alpha}, L1_ratio: {l1_ratio}, R-squared: {r_squared}")

print(f"\nBest Alpha: {best_alpha}")
print(f"Best L1_ratio: {best_l1_ratio}")
print(f"Best R-squared: {best_r2}")
print(f"Best coefficients: {best_beta}")

final_predictions = np.dot(X, best_beta)
print("\nFinal Predictions:", final_predictions)
results_df = pd.DataFrame({
    'Actual': y,
    'Predicted': final_predictions
})

plt.figure(figsize=(8, 6))
sns.regplot(x='Actual', y='Predicted', data=results_df, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.title('Linear Regression: Actual vs. Predicted Values')
plt.xlabel('Actual Prices (in Lacs)')
plt.ylabel('Predicted Prices (in Lacs)')
plt.show()
