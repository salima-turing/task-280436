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
    Calculate the Pearson correlation coefficient between two columns in a dataframe.

    Args:
        df (pandas.DataFrame): The input dataframe.
        x_col (str): Name of the column for the first variable.
        y_col (str): Name of the column for the second variable.

    Returns:
        float: The Pearson correlation coefficient.
    """
    corr, _ = stats.pearsonr(df[x_col], df[y_col])
    return corr


# Perform the correlation analysis
time_spent_col = 'Time_Spent_min'
quality_col = 'Content_Quality_Score'

correlation_coefficient = calculate_correlation(df, time_spent_col, quality_col)

print(f"Correlation between {time_spent_col} and {quality_col}: r = {correlation_coefficient:.2f}")

# Interpretation
if correlation_coefficient > 0:
    print("There is a positive linear relationship between time spent and content quality.")
elif correlation_coefficient < 0:
    print("There is a negative linear relationship between time spent and content quality.")
else:
    print("There is no linear relationship between time spent and content quality.")
