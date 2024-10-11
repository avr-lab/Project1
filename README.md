OVERVIEW:

The main goal of this project is to implement linear regression from the beginning using Python and its libraries such as Pandas, Numpy, Matplotlib, Seaborn and sciPy. 
The goal is to predict prices of houses using different features from the Paris housing dataset. 
Elastic net regression is a type of linear regression model that uses L1 and L2 (Lasso & Ridge) techniques. 

IMPLEMENTATION:

The ElasticNet model is a reversion design that applies two together L1 (Lasso) and L2 (Ridge) punishments while fitting the model. These penalties overfitting by  few of the model's coefficients and can even remove irrelevant features, answers to a more genralized solution
  
WHEN TO USE THIS MODEL?

1. When you need a regression model that demands regularization to prevent overfitting.
2. when the datasets holds multi collinearity that are very equated.
3. when you be going to select only specified feature from your dataset by decreasing the less useful ones.

TESTING THE MODEL:

The datasets used here in Paris Hosuong datasets to train and test the model. The working of models were given below,
1. Exploring the data- The first step was to chek the relationship between featuess and housing prices using a heat map.
2. Handling outliers- We capped the outliers at the 99th and 1st percentage to reduce the performance impact in the model. This tenchique is called as Winsorization.
3. Fitting the model- This model was trained by two parameters “alpha and l1_ratio”.
alpha : Controls the regularization.
l1_ratio : checks the balance between l1 and l2.
4. Evalutaions- This model was working with the R squaredmetric which indicates




PARAMETERS FOR IMPLEMENTATION:

1. alpha : Controls the regularization.
2. l1_ratio : checks the balance between l1(Lasso) and l2(Ridge).
	if l1_ratio = 1 ,  it goes to lasso.
	if l1_ratio = 0 ,  it goes to Ridge.
3. max_iter : specifies the maximum number of iterations allowed during the optimization process.
4. tol : sets the tolerance level for stopping the process. if changes it goes to tol, the optimization stops.

TROUBLES WITH IMPLEMENTATION:

Correlated Features: Although ElasticNet is planned to handle multicollinearity, well correlated features can still cause the same issues. In specific cases, further feature design may be necessary (for example, utilizing Principal Component Analysis or other range decline methods).

Outliers: Even after capping the extreme principles, outliers still unfavorably influence the model’s performance. In cases accompanying many outliers, further aberration situation (like log changing) concede possibility of to make regular the data.

Scalability: For huge datasets, this implementation concede possibility struggle to scale efficiently on account of the iterative optimization process. If scalability enhances an issue, more advanced growth thechniques or parallel processing must be considered.

OUR CODE:
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

Purpose: This portion imports the essential libraries for data manipulation, visualization, growth of model.

Df = pd.read_csv(r'C:\Users\vrani\Desktop\machine_learning\ParisHousing.csv')

Purpose: Load the dataset into pandas Data frame for post analysis.

df.head()
df.shape
df.columns
df.isnull().sum()

Purpose: By checking the dataset by displaying few columns and rows,  the figure of the data frame, the names of the columns and inspecting for some missing values. This step is important for understanding the data structure .

df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

Purpose: Filter for mathematical columns and create a correlation heatmap to visualize relation between the target and the variable. This helps in understanding which features are most appropriate for relevant prediction.

correlations = df.corr()['price'].sort_values(ascending=False)
print("Correlations with price:\n", correlations)

low_correlation_features = correlations[correlations.abs() < 0.1].index
df.drop(low_correlation_features, axis=1, inplace=True)

Purpose: Label and delete features accompanying low equivalence to the target changeable. This shortens the model and reduces making it easy to run and boost model performance.

for col in df.columns:
    if col != 'price':
        upper_bound = df[col].quantile(0.99)
        lower_bound = df[col].quantile(0.01)
        df[col] = np.clip(df[col], lower_bound, upper_bound)

Purpose: we initiate this step to stabilize the model and prevent extreme values from skewing predictions. we limited the value to the 1st and 99th percentilesto manage outliers.

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = (df[col] - mean) / std

Purpose: This one is known as critical for regression models because it ensures contribution equally to the model’s calculation. The numeric features are standardize so they have mean of 0 and constant deviation of 1.

X = df.drop('price', axis=1)
y = df['price']
X = sm.add_constant(X)

Purpose: We have divided the dataframe into features(x) and target variable. after that, we add a constant term which is necessary for the intercept in regression models.

def elastic_net_loss(beta, X, y, alpha=0.001, l1_ratio=0.5):
    ...
Purpose: We have defined a function name custom loss for elastic net regression . This functions helps to calculate loss based on predictions, penalities.

alphas = [0.0001, 0.001, 0.01, 0.1]  
l1_ratios = [0.1, 0.5, 0.9]  
best_r2 = -np.inf
...
Purpose: We loop through various combinations of regularization strengths (alpha) and L1/L2 ratios (l1_ratio) to find the best performing model based on the R-squared value. This is very important for optimizing the model's performance.

ss_total = np.sum((y - np.mean(y)) ** 2)  
ss_residual = np.sum((y - predictions) ** 2)  
r_squared = 1 - (ss_residual / ss_total)

Purpose: Here we calculate the R-squared values to evaluate the model's performance. The R-squared value indicates how well the model explains the variance in the target variable.

final_predictions = np.dot(X, best_beta)
print("\nFinal Predictions:", final_predictions)

Purpose: We make predictions using the optimized model coefficients derived from the best hyperparameters.

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

Purpose: We create a data frame to compare the actual and predicted values. Then we will visualize the results using a scatter plot graph with a regression line. This helps to visualize the model's performance.

