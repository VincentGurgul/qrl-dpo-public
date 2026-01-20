import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 11

df = pd.DataFrame({
    'Model': [
        'Equal Weights',
        'MVO',
        'RL with 27k params',
        'RL with 160k params',
        'QRL with 60 params',
    ],
    'Sharpe': [
        np.mean([0.4375]),
        np.mean([0.5919]),
        np.mean([ 0.4384,  0.4939]),
        np.mean([ 0.7926,  0.8237]),
        np.mean([ 0.7281,  0.6537])
    ],
    'Std': [
        np.mean([0.6558]),
        np.mean([0.9170]),
        np.mean([0.6915, 0.6905, 0.6621, 0.6623]),
        np.mean([0.6275, 0.6276, 0.5191, 0.5367]),
        np.mean([0.5475, 0.7395, 0.5574, 0.6155])
    ]
})

df = df.iloc[::-1].reset_index(drop=True)
y = np.arange(len(df)) * 1.5

plt.figure(figsize=(9, 5), dpi=300)
plt.barh(y, df['Sharpe'], height=1, alpha=0.7, label='Sharpe Ratio')

for i, (m, s) in enumerate(zip(df['Sharpe'], df['Std'])):
    plt.plot([m - s, m + s], [y[i], y[i]], color='black', linewidth=2, label='Std. Dev.' if i == 0 else None)

plt.yticks(y, df['Model'])
plt.axvline(0, linestyle='--', linewidth=1, color='black')
plt.xlabel('Sharpe Ratio', labelpad=12)
plt.legend()
plt.tight_layout()
plt.savefig('results_barplot.png')
