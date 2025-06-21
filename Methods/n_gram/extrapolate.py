import csv
import pandas as pd

data = pd.read_csv('data/sick/SICK.csv')

data_score = data['relatedness_score']
data['normalised_score'] = ((data_score - data_score.min()) / (data_score.max() - data_score.min())) * 5

data.to_csv('data/sick/SICK.csv', index=False)