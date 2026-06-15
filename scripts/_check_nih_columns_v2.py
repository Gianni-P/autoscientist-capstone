import pandas as pd
import os
from src.config import NIH_LABELS_CSV
df = pd.read_csv(NIH_LABELS_CSV)
print([f"'{c}'" for c in df.columns])
