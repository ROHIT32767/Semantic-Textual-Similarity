import sys

import numpy as np
import pandas as pd
import torch
from rnn_utils import load_data


def main():

    _,_,_,_,_,_,s1_test, s2_test, y_test = load_data()

    df_raw = pd.read_csv('../../test.csv')
    id = df_raw['id'].values
    s1_raw = df_raw['s1'].values
    s2_raw = df_raw['s2'].values
    y_raw = df_raw['score'].values

    model = torch.load(f'models/model.pt', weights_only=False)

    model.eval()

    with torch.no_grad():
        y_pred = model(s1_test, s2_test)
        y_pred = y_pred.numpy()
        y_pred = np.squeeze(y_pred)
        y_pred = np.clip(y_pred, 0, 5)
        results = pd.DataFrame({'id': id, 'sentence1': s1_raw, 'sentence2': s2_raw, 'similarity': y_raw, 'predicted_similarity': y_pred})
        results.to_csv(f'results/results_rnn.csv', index=False)
        print(f'Predictions saved in results_rnn.csv')

    correlation = pd.Series(y_raw).corr(pd.Series(y_pred))
    print('Correlation between expected and predicted similarity scores:', correlation)

if __name__ == '__main__':
    main()