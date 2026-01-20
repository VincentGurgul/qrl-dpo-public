import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utilities.data_processing import TimeSeriesCrossValidation

plt.rcParams['figure.dpi'] = 300

# --- Create fake data ---
N = 200
price_data = pd.DataFrame({'price': np.random.randn(N).cumsum()})

cv = TimeSeriesCrossValidation(price_data, n_splits=7)

segments = []   # store (fold, train_len, val_len, test_len)

for fold, (train_idx, test_idx) in enumerate(cv(), start=1):
    # 80/20 split inside train_idx → train/val
    val_split = int(len(train_idx) * 0.8)
    
    train_data = price_data.iloc[train_idx][:val_split]
    val_data   = price_data.iloc[train_idx][val_split:]
    test_data  = price_data.iloc[test_idx]

    segments.append([
        fold,
        len(train_data),
        len(val_data),
        len(test_data)
    ])

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 2 + 0.15 * len(segments)))

folds  = [f'Fold {s[0]}' for s in segments]
train_l = [s[1] for s in segments]
val_l   = [s[2] for s in segments]
test_l  = [s[3] for s in segments]

ax.barh(folds, train_l, label='Train')
ax.barh(folds, val_l, left=train_l, label='Validation')
ax.barh(folds, test_l, left=(np.array(train_l) + np.array(val_l)), label='Test')

# ax.set_title('Time Series Cross-Validation')
ax.invert_yaxis()
ax.legend(loc='upper right')

# Remove xticks + labels
ax.set_xticks([])
ax.set_xlabel('Time')

plt.tight_layout()
plt.savefig('time_cv_plot.png')
