import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
import matplotlib.pyplot as plt

# Dummy data
np.random.seed(0)
data = {
    'Time_Spent': np.random.randint(10, 1000, 100),
    'Content_Quality': np.random.randint(1, 10, 100),
    'Quiz_Score': np.random.randint(0, 100, 100),
    'Page_Views': np.random.randint(1, 1000, 100),
    'User_Type': np.random.choice(['Student', 'Teacher', 'Parent'], size=100)
}

df = pd.DataFrame(data)

# Calculate Pearson Correlation
corr_matrix = df.corr()
print("Correlation Matrix:")
print(corr_matrix)

# Let's explore the correlation between 'Time_Spent' and 'Content_Quality'
time_spent_quality_corr = pearsonr(df['Time_Spent'], df['Content_Quality'])
print("\nCorrelation between Time Spent and Content Quality: ", time_spent_quality_corr[0])

# Let's explore the correlation between 'Quiz_Score' and 'Content_Quality'
quiz_score_quality_corr = pearsonr(df['Quiz_Score'], df['Content_Quality'])
print("\nCorrelation between Quiz Score and Content Quality: ", quiz_score_quality_corr[0])

# To assess the significance of the correlations, we can use p-values
p_value = time_spent_quality_corr[1]
alpha = 0.05  # conventional significance level

if p_value > alpha:
    print("\nP-value:", p_value)
    print("The correlation between Time Spent and Content Quality is not significant (p>alpha).")
else:
    print("\nP-value:", p_value)
    print("The correlation between Time Spent and Content Quality is significant (p<=alpha).")

# Explore other statistical measures
print("\nSkewness of Time Spent: ", stats.skew(df['Time_Spent']))
print("Kurtosis of Time Spent: ", stats.kurtosis(df['Time_Spent']))

# Visualization - Scatter plot
plt.scatter(df['Time_Spent'], df['Content_Quality'])
plt.xlabel('Time Spent (in minutes)')
plt.ylabel('Content Quality')
plt.title('Scatter plot of Time Spent vs Content Quality')
plt.show()
