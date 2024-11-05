import pandas as pd
import numpy as np
from scipy import stats

### Create dummy data
np.random.seed(0)
num_users = 100
data = {
    'User_ID': np.arange(1, num_users + 1),
    'Time_Spent_min': np.random.randint(10, 100, size=num_users),
    'Content_Quality_Score': np.random.randint(1, 6, size=num_users),
}
df = pd.DataFrame(data)


def calculate_correlation(df, x_col, y_col):
    """
    Calculate the Pearson correlation coefficient and perform a t-test for correlation.

    Args:
        df (pandas.DataFrame): The input dataframe.
        x_col (str): Name of the column for the first variable.
        y_col (str): Name of the column for the second variable.

    Returns:
        tuple: A tuple containing the Pearson correlation coefficient and the p-value from the t-test.
    """
    corr, p_value = stats.pearsonr(df[x_col], df[y_col])
    return corr, p_value


def is_correlation_significant(p_value, alpha=0.05):
    """
    Interpret the p-value to determine if the correlation is statistically significant.

    Args:
        p_value (float): The p-value from the t-test.
        alpha (float): The significance level (default=0.05).

    Returns:
        bool: True if the correlation is statistically significant, False otherwise.
    """
    return p_value < alpha


# Perform the correlation analysis
time_spent_col = 'Time_Spent_min'
quality_col = 'Content_Quality_Score'

correlation_coefficient, p_value = calculate_correlation(df, time_spent_col, quality_col)

print(
    f"Correlation between {time_spent_col} and {quality_col}: r = {correlation_coefficient:.2f}, p-value = {p_value:.2f}")

# Interpretation
alpha = 0.05  # Significance level
if is_correlation_significant(p_value, alpha):
    print("The correlation is statistically significant at the α =", alpha, "level.")
else:
    print("The correlation is not statistically significant at the α =", alpha, "level.")
