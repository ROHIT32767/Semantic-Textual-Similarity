import pandas as pd
import matplotlib.pyplot as plt

def get_correlation(file):
    data = pd.read_csv(file)
    expected_scores = data['similarity'].values
    predicted_scores = data['predicted_similarity'].values
    correlation = pd.Series(expected_scores).corr(pd.Series(predicted_scores))
    return correlation
 
correlation_values = []
for n in range(1,6):
    # modify file path according to dataset
    correlation = get_correlation('results/result_n{}.csv'.format(n))
    correlation_values.append(correlation)
    print('Correlation for n = {}: {}'.format(n, correlation))

plt.plot(range(1,6), correlation_values, marker='o')
plt.xlabel('n')
plt.ylabel('Correlation')
plt.title('Correlation vs n')
plt.show()