"""Deck-quality figure for the Split CIFAR-100 continual-learning demo.

Reads results_cl_final.json / results_cl_joint.json (same dir) and renders:
  [left]  accuracy on all seen classes after each task (mean +/- std, 3 seeds)
  [right] two retention matrices (AdamW vs IVON-CL, seed 0)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
final = json.load(open(os.path.join(HERE, 'results_cl_final.json')))
joint = json.load(open(os.path.join(HERE, 'results_cl_joint.json')))

STYLE = {
    'adamw':     dict(color='#2a78d6', ls='--', lw=2.0, label='AdamW (naive)'),
    'adamw-ewc': dict(color='#eda100', ls=':',  lw=2.0, label='AdamW + EWC'),
    'ivon':      dict(color='#e87ba4', ls=':',  lw=2.0, label='IVON (naive)'),
    'ivon-cl':   dict(color='#0f9d6b', ls='-',  lw=3.2, label='IVON-CL (Bayes)'),
    'evon-cl':   dict(color='#4a3aa7', ls='--', lw=2.0, label='EVON-CL (Bayes)'),
}
ORDER = ['adamw', 'adamw-ewc', 'ivon', 'ivon-cl', 'evon-cl']

curves = {m: [] for m in ORDER}
Rmats = {}
for r in final['runs']:
    curves[r['method']].append([u['acc'] * 100 for u in r['union']])
    if r['seed'] == 0 and r['method'] in ('adamw', 'ivon-cl'):
        Rmats[r['method']] = r['R']
ceiling = max(r['metrics']['ACC'] for r in joint['runs']) * 100

fig = plt.figure(figsize=(14.4, 4.9), dpi=200)
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(1, 3, width_ratios=[2.25, 1, 1], wspace=0.30,
                       left=0.05, right=0.985, top=0.74, bottom=0.13)

# ---- left: forgetting curves ----
ax = fig.add_subplot(gs[0])
x = np.arange(1, 11)
SHORT = {'adamw': 'AdamW', 'adamw-ewc': 'AdamW+EWC', 'ivon': 'IVON naive',
         'ivon-cl': 'IVON-CL', 'evon-cl': 'EVON-CL'}
ends = []
for m in ORDER:
    arr = np.array(curves[m])
    mu, sd = arr.mean(0), arr.std(0)
    st = STYLE[m]
    ax.plot(x, mu, st['ls'], color=st['color'], lw=st['lw'], marker='o',
            ms=4.5 if m == 'ivon-cl' else 3.2, zorder=5 if m == 'ivon-cl' else 3)
    ax.fill_between(x, mu - sd, mu + sd, color=st['color'], alpha=0.13, lw=0)
    ends.append([m, mu[-1], mu[-1]])           # (method, true y, label y)
# dodge overlapping end labels (min 4.2 pts of separation, top-down)
ends.sort(key=lambda e: -e[1])
for i in range(1, len(ends)):
    if ends[i - 1][2] - ends[i][2] < 4.2:
        ends[i][2] = ends[i - 1][2] - 4.2
for m, y_true, y_lab in ends:
    st = STYLE[m]
    ax.annotate(f"{SHORT[m]}  {y_true:.1f}%", (10.25, y_lab),
                color=st['color'], fontsize=10, va='center',
                fontweight='bold' if m.endswith('-cl') else 'normal',
                annotation_clip=False)
    if abs(y_lab - y_true) > 0.5:
        ax.plot([10.05, 10.22], [y_true, y_lab], color=st['color'],
                lw=0.7, alpha=0.6, clip_on=False)
ax.axhline(ceiling, color='#9a9890', lw=1.2, ls=(0, (2, 4)))
ax.annotate(f'joint-training ceiling  {ceiling:.1f}%', (1.0, ceiling + 1.4),
            color='#807e76', fontsize=9)
ax.set_xlim(0.8, 10.1); ax.set_ylim(20, 102)
plt.setp(ax, xmargin=0)
fig.subplots_adjust(left=0.05)
box = ax.get_position(); ax.set_position([box.x0, box.y0, box.width * 0.82, box.height])
ax.set_xticks(x); ax.set_xlabel('tasks learned sequentially', fontsize=11)
ax.set_ylabel('accuracy on all seen classes (%)', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#eceae4', lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(labelsize=9.5)

# ---- right: retention matrices ----
def heat(axh, R, title, accent):
    M = np.full((10, 10), np.nan)
    for t, row in enumerate(R):
        M[t, :len(row)] = np.array(row) * 100
    im = axh.imshow(M, cmap='Greens', vmin=0, vmax=100)
    for t in range(10):
        for j in range(t + 1):
            v = M[t, j]
            axh.text(j, t, f'{v:.0f}', ha='center', va='center',
                     fontsize=6.4, color='white' if v > 55 else '#44423c')
    axh.set_title(title, fontsize=10.5, color=accent, pad=8, fontweight='bold')
    axh.set_xlabel('evaluated task', fontsize=9)
    axh.set_xticks(range(10), [str(i + 1) for i in range(10)], fontsize=7)
    axh.set_yticks(range(10), [str(i + 1) for i in range(10)], fontsize=7)
    axh.spines[:].set_visible(False)
    axh.tick_params(length=0)

axA = fig.add_subplot(gs[1])
heat(axA, Rmats['adamw'], 'AdamW — task 1 is erased', '#2a78d6')
axA.set_ylabel('after training task…', fontsize=9)
axB = fig.add_subplot(gs[2])
heat(axB, Rmats['ivon-cl'], 'IVON-CL — early tasks survive', '#0f9d6b')

fig.suptitle('Continual learning without replay: the optimizer’s posterior is the memory',
             fontsize=16, x=0.05, ha='left', y=0.975, fontweight='bold', color='#232320')
fig.text(0.05, 0.885,
         'Split CIFAR-100 · 10 tasks × 10 classes, class-incremental, replay-free · linear head on frozen DINOv2 ViT-S/14 features · mean ± std, 3 seeds',
         fontsize=9.6, color='#6b6963', va='top')
fig.text(0.05, 0.838,
         'IVON-CL / EVON-CL: each task’s posterior — mean and precision read from the blrax optimizer state — becomes the prior for the next task',
         fontsize=9.6, color='#6b6963', va='top')

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'cl_demo_figure.png')
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print('wrote', out)
