import pandas as pd
import numpy as np
from scipy import stats

### Create dummy data
np.random.seed(0)
num_users = 10000
data = {
    'User_ID': np.arange(1, num_users + 1),
    'Time_Spent_min': np.random.randint(10, 200, size=num_users),
    'Content_Quality_Score': np.random.randint(1, 6, size=num_users),
}

df = pd.DataFrame(data)


def calculate_correlation(df, x_col, y_col):
    """
    Calculate the Pearson correlation coefficient and its significance.

    Args:
        df (pandas.DataFrame): The input dataframe.
        x_col (str): Name of the column for the first variable.
        y_col (str): Name of the column for the second variable.

    Returns:
        corr: The Pearson correlation coefficient.
        p_value: The p-value associated with the correlation.
        conf_int: The 95% confidence interval for the correlation coefficient.
    """
    corr, p_value = stats.pearsonr(df[x_col], df[y_col])
    conf_int = stats.pearsonr_ci(df[x_col], df[y_col], conf_level=0.95)
    return corr, p_value, conf_int


# Perform the correlation analysis
time_spent_col = 'Time_Spent_min'
quality_col = 'Content_Quality_Score'

correlation_coefficient, p_value, conf_int = calculate_correlation(df, time_spent_col, quality_col)

print(f"Correlation between {time_spent_col} and {quality_col}: r = {correlation_coefficient:.4f}")
print(f"P-value: p = {p_value:.4f}")
print(f"95% Confidence Interval: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")

# Interpretation
# Statistical significance is indicated by a p-value < 0.05
alpha = 0.05
if p_value < alpha:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")
