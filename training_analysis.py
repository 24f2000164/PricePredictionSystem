# ============================================================
# TRAINING ANALYSIS — Load history.json and plot everything
# Run this AFTER MLPrice_improved.py completes
# ============================================================

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HISTORY_PATH = r'C:\Users\kumar\Desktop\price_prediction\history.json'
SAVE_PATH    = r'C:\Users\kumar\Desktop\price_prediction\final_analysis.png'

with open(HISTORY_PATH) as f:
    h = json.load(f)

epochs = list(range(1, len(h['train_loss']) + 1))

fig = plt.figure(figsize=(22, 14), facecolor='#0d1117')
fig.suptitle('Full Training Analysis Report', fontsize=18, color='white', y=0.99)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)
ax  = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]

def sax(a, title, xl='Epoch', yl=''):
    a.set_facecolor('#161b22')
    a.set_title(title, color='#e6edf3', fontsize=11, pad=8)
    a.set_xlabel(xl, color='#8b949e', fontsize=9)
    a.set_ylabel(yl, color='#8b949e', fontsize=9)
    a.tick_params(colors='#8b949e')
    for sp in a.spines.values(): sp.set_edgecolor('#30363d')
    a.grid(True, color='#21262d', linestyle='--', linewidth=0.7)

# 0: Loss curves
sax(ax[0], 'Train vs Val Loss', yl='Loss')
ax[0].plot(epochs, h['train_loss'], color='#58a6ff', marker='o', label='Train')
ax[0].plot(epochs, h['val_loss'],   color='#f78166', marker='s', label='Val')
ax[0].legend(facecolor='#21262d', labelcolor='white', fontsize=9)

# 1: SMAPE trend + target line
sax(ax[1], 'Val SMAPE over Epochs', yl='SMAPE %')
ax[1].plot(epochs, h['val_smape'], color='#3fb950', marker='D', linewidth=2.5)
best_i = int(np.argmin(h['val_smape']))
ax[1].scatter([epochs[best_i]], [h['val_smape'][best_i]], color='gold', zorder=6, s=150,
              label=f"Best: {h['val_smape'][best_i]:.2f}%")
ax[1].axhline(34,  color='#f78166', ls=':',  lw=1.5, label='Target 34%')
ax[1].axhline(36.5,color='#e3b341', ls='--', lw=1,   label='Baseline 36.5%')
ax[1].legend(facecolor='#21262d', labelcolor='white', fontsize=8)

# 2: Overfitting gap
sax(ax[2], 'Overfitting Gap (Val − Train)', yl='Gap')
gap   = [v - t for v, t in zip(h['val_loss'], h['train_loss'])]
cols  = ['#f78166' if g > 0.15 else '#3fb950' for g in gap]
ax[2].bar(epochs, gap, color=cols)
ax[2].axhline(0.15, color='#e3b341', ls='--', lw=1, label='Warn: 0.15')
ax[2].legend(facecolor='#21262d', labelcolor='white', fontsize=8)

# 3: MAE over epochs
sax(ax[3], 'Val MAE ($)', yl='$')
ax[3].plot(epochs, h['val_mae'], color='#ffa657', marker='v', linewidth=2)
ax[3].fill_between(epochs, h['val_mae'], alpha=0.15, color='#ffa657')

# 4: LR curve (head)
sax(ax[4], 'Head Learning Rate', yl='LR')
ax[4].plot(epochs, h['lr_head'], color='#d2a8ff', marker='^')
ax[4].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax[4].yaxis.get_offset_text().set_color('#8b949e')

# 5: Gradient norm
sax(ax[5], 'Mean Gradient Norm', yl='Norm')
ax[5].plot(epochs, h['grad_norm'], color='#79c0ff', marker='o')
ax[5].axhline(0.5, color='#f78166', ls='--', lw=1, label='Clip=0.5')
ax[5].legend(facecolor='#21262d', labelcolor='white', fontsize=8)

# 6-8: Per-segment SMAPE across epochs (one line per segment)
segs_over_time = {k: [] for k in h['segment_smape'][0].keys()}
for epoch_segs in h['segment_smape']:
    for k, v in epoch_segs.items():
        segs_over_time[k].append(v)

seg_colors = {'<$10': '#58a6ff', '$10-$30': '#3fb950', '$30-$100': '#e3b341', '>$100': '#f78166'}
sax(ax[6], 'Per-Segment SMAPE over Epochs', yl='SMAPE %')
for seg, vals in segs_over_time.items():
    ax[6].plot(epochs, vals, marker='o', label=seg, color=seg_colors.get(seg, 'white'))
ax[6].legend(facecolor='#21262d', labelcolor='white', fontsize=8)

# 7: Final epoch bar breakdown
sax(ax[7], f'Per-Segment SMAPE — Final Epoch {epochs[-1]}', yl='SMAPE %')
last_segs = h['segment_smape'][-1]
labs = list(last_segs.keys()); vals = [last_segs[k] for k in labs]
bar_c = [seg_colors.get(l, '#8b949e') for l in labs]
bars  = ax[7].bar(labs, vals, color=bar_c)
for bar, val in zip(bars, vals):
    ax[7].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=9)

# 8: Improvement from baseline
sax(ax[8], 'SMAPE Improvement vs Baseline (36.5%)', yl='Δ SMAPE pp')
improvement = [36.5 - s for s in h['val_smape']]
cols2 = ['#3fb950' if i > 0 else '#f78166' for i in improvement]
ax[8].bar(epochs, improvement, color=cols2)
ax[8].axhline(0, color='white', lw=0.8)
ax[8].axhline(2.5, color='gold', ls='--', lw=1, label='Target: −2.5 pp')
ax[8].legend(facecolor='#21262d', labelcolor='white', fontsize=8)

plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"Final analysis saved → {SAVE_PATH}")

# ── text summary ──────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
best_smape = min(h['val_smape'])
best_epoch = h['val_smape'].index(best_smape) + 1
print(f"Best SMAPE  : {best_smape:.2f}%  (epoch {best_epoch})")
print(f"Best MAE    : ${min(h['val_mae']):.2f}")
print(f"Max gap     : {max(g := [v-t for v,t in zip(h['val_loss'], h['train_loss'])]):.4f}  "
      f"(epoch {g.index(max(g))+1})")
print(f"Final LR    : {h['lr_head'][-1]:.2e}")

print("\nFinal-epoch segment SMAPE:")
for seg, sv in h['segment_smape'][-1].items():
    print(f"  {seg:>10s}: {sv:.2f}%")

if best_smape < 34:
    print(f"\n TARGET ACHIEVED: {best_smape:.2f}% < 34%")
else:
    remaining = best_smape - 34
    print(f"\n {remaining:.2f} pp remaining to reach 34% target")
 