import pandas as pd
df = pd.read_csv('data/PadChest/padchest_meta.csv')
print(df['Labels'].dropna().head(20).tolist())
