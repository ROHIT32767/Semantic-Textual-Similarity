from score import get_similarity_score
import pandas as pd
from tqdm import tqdm

data = pd.read_csv('../../test.csv')

# modify according to dataset
sentences_A = data['s1'].values
sentences_B = data['s2'].values
scores = data['score'].values

n_values = [1,2,3,4,5]
for n in n_values:
    print('Predicting for n = {}'.format(n))
    predicted_scores = []
    sent1 = []
    sent2 = []
    exp_scores = []
    for i in tqdm(range(len(sentences_A))):
        predicted_score = get_similarity_score(sentences_A[i], sentences_B[i],n)
        if predicted_score == -1:
            continue
        predicted_scores.append(predicted_score)
        sent1.append(sentences_A[i])
        sent2.append(sentences_B[i])
        exp_scores.append(scores[i])

    results = pd.DataFrame({'sentence1': sent1, 'sentence2': sent2 ,'similarity': exp_scores, 'predicted_similarity': predicted_scores})
    results.to_csv('results/result_n{}.csv'.format(n), index=False)