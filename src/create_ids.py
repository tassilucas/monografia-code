import pandas as pd
import numpy as np

df = pd.read_csv('../dataset/articles.csv')
df['ID'] = np.arange(df.shape[0])
df.to_csv('../dataset/articles_with_id.csv', index=False)
