"""Numbers for the M3 tables of the report, computed from the result files in ../results.

Run from this folder:   python make_tables.py

It prints the LaTeX rows for Table 6 (RQ2 edits) and Table 7 (RQ3 detectors). Paste them into the
tabular in main.tex. make_figures.py uses the same numbers for the figures.

Where each table comes from:
  Table 6  ../results/setup_check_meta.json (PNG, JPEG, resize), rq2_screenshot_per_photo.csv,
           rq2_vae_per_photo.csv, rq2_inpainting_per_photo.csv (composited rows)
  Table 7  ../results/rq3_detectors_summary.csv
  Table 5  the training notebooks in ../../m3_watermark (their last test cells write test_summary.csv in the
           results zips on the team Drive). It is not computed from files in this repo.
"""
import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / 'results'


def _rows(name):
    with open(RESULTS / name, newline='') as f:
        return list(csv.DictReader(f))


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def _is_true(value):
    return str(value) == 'True'


def _per_photo(name, **match):
    rows = [r for r in _rows(name) if all(r[k] == v for k, v in match.items())]
    return (100 * _mean(float(r['bit_accuracy']) for r in rows),
            100 * _mean(_is_true(r['valid_verdict']) for r in rows), len(rows))


def rq2_numbers():
    """Rows of Table 6 as (type, edit, bit accuracy %, valid verdict %), in the order of the table."""
    meta = {r['edit']: r for r in json.loads((RESULTS / 'setup_check_meta.json').read_text())['table']}
    rows = [('--', 'Saved as PNG', meta['PNG']), ('Normal', 'JPEG quality 90', meta['JPEG 90']),
            ('Normal', 'JPEG quality 70', meta['JPEG 70']), ('Normal', 'JPEG quality 50', meta['JPEG 50']),
            ('Normal', 'Resized to 224\\,px', meta['Resized to 224'])]
    out = [(kind, edit, float(m['bit_accuracy_%']), float(m['valid_verdict_%'])) for kind, edit, m in rows]
    bits, valid, _ = _per_photo('rq2_screenshot_per_photo.csv', edit='Screenshot simulation (all steps)',
                                source='stamped')
    out.append(('Normal', 'Screenshot', bits, valid))
    bits, valid, _ = _per_photo('rq2_vae_per_photo.csv', edit='VAE round trip', source='stamped')
    out.append(('AI', 'VAE round trip', bits, valid))
    for percent in (10, 25, 50):
        bits, valid, _ = _per_photo('rq2_inpainting_per_photo.csv', edit=f'Inpainting, {percent}% mask',
                                    variant='composited')
        out.append(('AI', f'Inpainting, {percent}\\%', bits, valid))
    return out


def rq3_numbers():
    """Rows of Table 7 as dictionaries, from rq3_detectors_summary.csv."""
    names = {'m1_resnet': 'M1', 'm2_clip': 'M2', 'fusion': 'Late Fusion'}
    return [{'name': names[r['detector']], **{k: float(v) for k, v in r.items() if k != 'detector'}}
            for r in _rows('rq3_detectors_summary.csv')]


def _signed(x):
    return f'${x:+.3f}$'


def _bound(x):
    return f'${x:.3f}$'


def table6_latex():
    return '\n'.join(f'{kind} & {edit} & {bits:.2f}\\% & {valid:.1f}\\% \\\\'
                     for kind, edit, bits, valid in rq2_numbers())


def table7_latex():
    lines = []
    for r in rq3_numbers():
        lines.append(f"{r['name']} & {_signed(r['score_change_real'])} [{_bound(r['ci95_real_low'])}, "
                     f"{_bound(r['ci95_real_high'])}] & {_signed(r['score_change_ai'])} "
                     f"[{_bound(r['ci95_ai_low'])}, {_bound(r['ci95_ai_high'])}] & {int(r['verdict_flips'])} & "
                     f"{r['accuracy_before']:.3f} / {r['accuracy_after']:.3f} & "
                     f"{r['auroc_before']:.3f} / {r['auroc_after']:.3f} \\\\")
    return '\n'.join(lines)


if __name__ == '__main__':
    print('% Table 6 rows (Type & Edit & Bit accuracy & Valid verdict)')
    print(table6_latex())
    print('\n% Table 7 rows (Detector & Change on Real [CI] & Change on AI [CI] & Verdicts changed & Accuracy & AUROC)')
    print(table7_latex())
