import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame 
import nltk
df = pd.read_json("1.json", lines=True)
df['helpful'] = df['helpful'].map(lambda x: str(x)[:-1])
df['helpful'] = df['helpful'].map(lambda x: str(x)[1:])
df['HelpfulnessNumerator'],df['HelpfulnessDenominator'] = df['helpful'].apply(lambda x: pd.Series(x.split(', ')))
del df['helpful']
df.to_csv("reviews.csv", sep=',', index=False)
print(df.columns)
print(df.shape)
