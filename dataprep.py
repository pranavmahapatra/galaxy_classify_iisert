import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data filtering
rawData = (
    pd.read_csv('data.csv')
      .replace(r'^\s*$', np.nan, regex=True)
)

data_r = rawData[rawData.iloc[:, :-1].ne(0).all(axis=1)]
data_r = data_r.dropna()

data_r.to_csv('filtered_data.csv', index=False)

first_col = rawData.columns[0]
last_col  = rawData.columns[-1]
feature_cols = rawData.columns[1:-1]

# correct boolean mask + counts
zero_counts = ((rawData[feature_cols] == 0) | (rawData[feature_cols].isna())).sum()

# keeping columns with at most 1 zero/NaN
keep_features = zero_counts[zero_counts == 0].index

data_c = pd.concat([rawData[first_col],rawData[keep_features],rawData[last_col]], axis=1)
data_c = data_c.dropna(axis=1)

data_c.to_csv('filtered_data2.csv', index=False)

print("\ndata_f: data filtered by eliminating rows")
print("Original shape:", rawData.shape)
print("New shape:", data_r.shape)

print("\ndata_n: data filtered by eliminating columns")
print("Original shape:", rawData.shape)
print("New shape:", data_c.shape)
print("Dropped columns:", set(feature_cols) - set(keep_features))