"""PermutedMNIST figure: avg accuracy over seen tasks, 10 tasks, 2x100 MLP,
single head, replay-free — ours vs published anchors."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = np.arange(1, 11)
CUR = {
 'scratch-merge': ([97.9,42.8,22.7,21.1,17.4,14.7,14.1,13.8,12.2,12.5], '#b3b1a9', ':', 2.0,
                   'per-task posteriors merged (BMR-style)  12.5%'),
 'adamw':        ([97.7,82.6,67.2,56.4,49.3,44.9,40.5,35.5,31.3,30.6], '#2a78d6', '--', 2.0,
                  'AdamW naive  30.6%'),
 'ivon':         ([97.9,87.5,76.6,70.2,65.8,58.7,52.5,47.1,43.0,41.0], '#e87ba4', ':', 2.0,
                  'IVON naive  41.0%'),
 'adamw-ewc':    ([97.7,93.4,89.9,87.1,83.5,74.8,69.3,66.3,62.6,58.9], '#eda100', ':', 2.0,
                  'AdamW+EWC  58.9%'),
 'evon-cl':      ([97.8,90.6,87.4,85.3,83.9,82.2,80.4,79.8,79.1,79.0], '#4a3aa7', '--', 2.0,
                  'EVON-CL  79.0%'),
 'ivon-cl':      ([97.8,90.6,87.4,85.4,84.0,82.5,81.5,80.9,79.8,79.5], '#0f9d6b', '-', 3.2,
                  'IVON-CL  79.5%'),
 'covon-style':  ([97.5,95.8,93.2,91.8,89.2,86.0,83.5,82.3,79.3,79.9], '#0f9d6b', (0,(4,2)), 2.0,
                  'IVON-CL (CoVON lr sched)  79.9%'),
}
ANCHORS = [('VCL', 78, '(Nguyen et al. 18; -Hard, Melo et al.)'),
           ('VCL+coreset', 81, ''), ('UCB', 83, ''),
           ('TD-VCL', 89, '(Melo et al. 25)'),
           ('CoVON', 92.1, '(IVON-based, arXiv 2606.24007)')]

fig, ax = plt.subplots(figsize=(12.8, 5.4), dpi=200)
fig.patch.set_facecolor('white')
ends = []
for k, (c, col, ls, lw, lab) in CUR.items():
    ax.plot(x, c, ls=ls, color=col, lw=lw, marker='o',
            ms=4.5 if k == 'ivon-cl' else 3.0, zorder=5 if 'cl' in k else 3)
    ends.append([col, lab, c[-1], c[-1], 'cl' in k and k != 'scratch-merge'])
ends.sort(key=lambda e: -e[2])
for i in range(1, len(ends)):
    if ends[i-1][3] - ends[i][3] < 3.6:
        ends[i][3] = ends[i-1][3] - 3.6
for col, lab, y0, y1, bold in ends:
    ax.annotate(lab, (10.2, y1), color=col, fontsize=9.6, va='center',
                fontweight='bold' if bold else 'normal', annotation_clip=False)
    if abs(y1-y0) > .5:
        ax.plot([10.03, 10.17], [y0, y1], color=col, lw=.7, alpha=.6, clip_on=False)
for name, v, src in ANCHORS:
    ax.scatter([10], [v], marker='D', s=26, facecolor='none',
               edgecolor='#8a8880', lw=1.2, zorder=6)
    ax.annotate(f'{name} {v:g} {src}', (9.93, v), color='#807e76', fontsize=8,
                ha='right', va='center', style='italic')
ax.axhline(10, color='#c9c7bf', lw=1, ls=(0, (2, 4)))
ax.annotate('chance', (1.0, 11), color='#a09e96', fontsize=8.5)
ax.set_xlim(0.85, 10.1); ax.set_ylim(5, 101)
ax.set_xticks(x)
ax.set_xlabel('tasks learned sequentially', fontsize=11.5)
ax.set_ylabel('avg accuracy on all seen tasks (%)', fontsize=11.5)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#eceae4', lw=0.8); ax.set_axisbelow(True)
box = ax.get_position(); ax.set_position([box.x0, box.y0, box.width*0.78, box.height*0.88])
fig.suptitle('PermutedMNIST (10 tasks, 2×100 MLP, single head, replay-free): recursion works, merging does not',
             fontsize=14.5, x=0.06, ha='left', y=0.97, fontweight='bold', color='#232320')
fig.text(0.06, 0.895, 'same IVON posteriors in both: merged per-parameter across independent runs → chance (weight-permutation symmetry) · '
         'used recursively as sequential priors → 79.5–79.9%, above published VCL (78)',
         fontsize=9.4, color='#6b6963', va='top')
fig.savefig(__file__.replace('make_pmnist_figure.py', 'pmnist_figure.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
print('wrote pmnist_figure.png')
