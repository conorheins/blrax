"""Grouped-bar summary: final ACC by method across the four CL benchmarks."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BENCH = ['Split CIFAR-100\n(probe, CIL)', 'Rotated CIFAR-10\n(probe, DIL)',
         'Split ImageNet-R\n(probe, CIL)', 'Split CIFAR-100\n(LoRA backbone, CIL)']
METHODS = ['AdamW', 'AdamW+EWC', 'IVON naive', 'IVON-CL', 'EVON-CL']
COL = ['#2a78d6', '#eda100', '#e87ba4', '#0f9d6b', '#4a3aa7']
ACC = np.array([
    [29.3, 58.5, 54.3, 72.5, 72.6],
    [82.1, 84.6, 85.0, 85.2, 85.1],
    [18.4, 54.2, 44.8, 57.9, 58.1],
    [8.6, np.nan, np.nan, 22.8, 22.3],
])
ERR = np.array([
    [0.6, 1.7, 2.0, 0.8, 0.9],
    [0.4, 1.4, 0.3, 0.5, 0.5],
    [0.4, 1.5, 1.3, 0.7, 0.9],
    [np.nan]*5,
])
CEIL = [87.6, 84.9, 74.7, 86.8]

fig, ax = plt.subplots(figsize=(12.6, 4.4), dpi=200)
fig.patch.set_facecolor('white')
x = np.arange(4); w = 0.16
for i, m in enumerate(METHODS):
    vals = ACC[:, i]
    ax.bar(x + (i - 2) * w, vals, w * 0.92, color=COL[i], label=m,
           yerr=np.nan_to_num(ERR[:, i]), error_kw=dict(lw=1, capsize=2, ecolor='#555'))
    for b, v in zip(x + (i - 2) * w, vals):
        if not np.isnan(v):
            ax.text(b, v + 2.2, f'{v:.0f}', ha='center', fontsize=8.2,
                    color=COL[i], fontweight='bold' if m.endswith('-CL') else 'normal')
for xi, c in zip(x, CEIL):
    ax.plot([xi - 2.9*w, xi + 2.9*w], [c, c], color='#8a8880', lw=1.3, ls=(0, (2, 3)))
    ax.text(xi + 2.9*w, c + 1.2, f'joint {c:.0f}', ha='right', fontsize=8, color='#807e76')
ax.set_xticks(x, BENCH, fontsize=10)
ax.set_ylabel('final accuracy, all seen classes (%)', fontsize=11)
ax.set_ylim(0, 100)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#eceae4', lw=0.8); ax.set_axisbelow(True)
ax.legend(ncol=5, frameon=False, fontsize=9.5, loc='upper center',
          bbox_to_anchor=(0.5, 1.13))
fig.suptitle('Bayesian continual learning across four benchmarks (replay-free, frozen or LoRA-adapted DINOv2)',
             fontsize=13.5, x=0.05, ha='left', y=1.06, fontweight='bold', color='#232320')
fig.savefig(__file__.replace('make_cl_summary_fig.py', 'cl_summary_figure.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
print('wrote cl_summary_figure.png')
