"""The four M3 figures of the report, drawn from the result files in ../results (vector PDF, small files).

Run from this folder:   python make_figures.py          (needs matplotlib and numpy)
The PDFs are written to ./figs. The same PDFs are committed, so they can also be downloaded directly.

Where the numbers come from:
  Fig. 1  the pipeline steps; the thresholds are read from ../results/pipeline_meta.json
  Fig. 2  make_tables.rq2_numbers() (Table 6)
  Fig. 3  full size: the first four rows of Table 6 (PNG, JPEG 90, 70, 50).
          128 px: the "jpeg_resize_best_epoch97" rows of test_summary.csv in results_3_final_model.zip
          (team Drive), typed below (bit accuracy, and valid_correct_verdict as the valid verdict).
  Fig. 4  ../results/rq3_detectors_per_photo.csv
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so make_tables is found
import make_tables

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
OUT = HERE / 'figs'
OUT.mkdir(exist_ok=True)

# Okabe-Ito colours (colour-blind safe)
BLUE, ORANGE, GREEN, RED, GREY = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#666666'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})

rq2 = make_tables.rq2_numbers()          # (type, edit, bit accuracy %, valid verdict %) in the order of Table 6

# ---------------------------------------------------------------- Fig. 2: RQ2, bit accuracy and valid verdict by edit
short = ['PNG', 'JPEG 90', 'JPEG 70', 'JPEG 50', 'Resize 224', 'Screenshot', 'VAE round trip',
         'Inpaint 10%', 'Inpaint 25%', 'Inpaint 50%']
bits = [r[2] for r in rq2]
valid = [r[3] for r in rq2]
x = np.arange(len(short))
w = 0.38
fig, ax = plt.subplots(figsize=(3.45, 2.7))
ax.axvspan(5.5, 9.5, color='#EEEEEE', zorder=0)
ax.bar(x - w / 2, bits, w, color=BLUE, label='Bit accuracy')
ax.bar(x + w / 2, valid, w, color=ORANGE, label='Valid verdict')
ax.axhline(50, color=GREY, lw=0.7, ls='--', label='Guessing (50% of bits)')
ax.set_xticks(x)
ax.set_xticklabels(short, rotation=40, ha='right', rotation_mode='anchor')
ax.set_ylim(0, 132)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel('%')
ax.text(2.5, 103, 'Normal edits', ha='center', fontsize=7)
ax.text(7.5, 103, 'AI edits', ha='center', fontsize=7)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=3, fontsize=6.3,
          columnspacing=0.8, handlelength=1.4)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / 'fig_rq2_edits.pdf')
plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: JPEG, full size versus 128 px
labels = ['PNG', 'JPEG 90', 'JPEG 70', 'JPEG 50']
full_bits, full_valid = bits[:4], valid[:4]
small_bits, small_valid = [99.98, 98.24, 92.08, 86.24], [99.2, 57.2, 8.8, 1.2]     # 128 px, from the Drive zip
fig, ax = plt.subplots(figsize=(3.45, 2.4))
xs = np.arange(4)
ax.plot(xs, full_valid, '-o', color=BLUE, ms=3, label='Valid verdict, full size')
ax.plot(xs, small_valid, '-s', color=RED, ms=3, label='Valid verdict, 128 px')
ax.plot(xs, full_bits, '--o', color=BLUE, ms=3, mfc='white', label='Bit accuracy, full size')
ax.plot(xs, small_bits, '--s', color=RED, ms=3, mfc='white', label='Bit accuracy, 128 px')
ax.set_xticks(xs)
ax.set_xticklabels(labels)
ax.set_ylim(0, 105)
ax.set_ylabel('%')
ax.legend(loc='lower left', frameon=False, fontsize=6.5)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / 'fig_jpeg_fullsize_vs_128.pdf')
plt.close(fig)

# ---------------------------------------------------------------- Fig. 4: RQ3, change in P(AI) when a photo is stamped
with open(RESULTS / 'rq3_detectors_per_photo.csv', newline='') as f:
    rows = list(csv.DictReader(f))
names = [('m1_resnet', 'M1'), ('m2_clip', 'M2'), ('fusion', 'Late Fusion')]
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1))
bins = np.linspace(-0.7, 1.0, 52)        # covers every photo (smallest change -0.631, largest +0.934)
for ax, (key, title) in zip(axes, names):
    rs = [r for r in rows if r['detector'] == key]
    for label, name, colour in (('0', 'Real', BLUE), ('1', 'AI', ORANGE)):
        d = [float(r['p_after']) - float(r['p_before']) for r in rs if r['label'] == label]
        ax.hist(d, bins=bins, alpha=0.65, color=colour, label=f'{name} photos, mean {np.mean(d):+.3f}')
    ax.axvline(0, color='k', lw=0.6)
    ax.set_title(title)
    ax.set_xlabel('P(AI) after stamping minus before')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(frameon=False, fontsize=6, loc='upper right')
axes[0].set_ylabel('Photos')
fig.tight_layout(pad=0.3)
fig.savefig(OUT / 'fig_rq3_score_change.pdf')
plt.close(fig)

# ---------------------------------------------------------------- Fig. 1: the pipeline diagram
meta = json.loads((RESULTS / 'pipeline_meta.json').read_text())
t_low, t_high = meta['thresholds']
fig, ax = plt.subplots(figsize=(7.1, 2.35))
ax.set_xlim(0, 10.15)
ax.set_ylim(0, 3.45)
ax.axis('off')
LIGHT = {'m3': '#CFE3F3', 'det': '#FBE3B5', 'io': '#E6E6E6'}
EDGE = {'m3': BLUE, 'det': ORANGE, 'io': GREY}


def box(cx, cy, bw, bh, text, kind):
    ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=LIGHT[kind], ec=EDGE[kind], lw=1.0))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=6.4)


def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color='#333333', lw=0.9, shrinkA=0, shrinkB=0))


top, bottom = 2.65, 0.8
box(0.8, top, 1.3, 0.8, 'Input photo', 'io')
box(2.55, top, 1.7, 0.8, 'Read the stamp\n(M3 decoder)', 'm3')
cx, dw, dh = 4.55, 1.7, 1.1
ax.add_patch(Polygon([(cx - dw / 2, top), (cx, top + dh / 2), (cx + dw / 2, top), (cx, top - dh / 2)],
                     closed=True, fc='#FFFFFF', ec=EDGE['m3'], lw=1.0))
ax.text(cx, top, 'Checksum\nvalid?', ha='center', va='center', fontsize=6.4)
box(7.95, top, 3.3, 0.8, 'Return the stored verdict\n(the detectors are skipped)', 'io')
box(4.55, bottom, 1.9, 1.05, 'M1 and M2 score\nthe photo; Late Fusion\naverages them', 'det')
box(6.8, bottom, 1.9, 1.05, f'Thresholds {t_low:.3f}\nand {t_high:.3f} give\nReal, Uncertain or AI', 'det')
box(9.05, bottom, 1.9, 1.05, 'M3 stamps the\nverdict and returns\nthe stamped photo', 'm3')
arrow(1.45, top, 1.7, top)
arrow(3.4, top, 3.7, top)
arrow(cx + dw / 2, top, 6.3, top)
ax.text(5.95, top + 0.13, 'yes', fontsize=6.4, ha='center')
arrow(cx, top - dh / 2, cx, bottom + 0.525)
ax.text(cx + 0.12, 1.77, 'no', fontsize=6.4, ha='left')
arrow(5.5, bottom, 5.85, bottom)
arrow(7.75, bottom, 8.1, bottom)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / 'fig_pipeline.pdf')
plt.close(fig)

print('Wrote', sorted(p.name for p in OUT.glob('*.pdf')))
