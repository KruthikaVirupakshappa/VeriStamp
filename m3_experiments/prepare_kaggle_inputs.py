"""Build the Kaggle input dataset for the M3 notebooks (veristamp-m3-inputs) and check it.

Laptop (files you already have), then upload the zip on Kaggle as a private dataset:

    python prepare_kaggle_inputs.py out/veristamp-m3-inputs --zip \
        --m3-model "VeriStamp M3 results.zip"      # the Drive download, results_3_final_model.zip or best.pt
        --test-dir hidden_data/test                # the 500 test photos from Rishitha
        [--m1-checkpoint best.pt --m1-vslib vslib_m1_final]
        [--genimage-splits <split folder> --genimage-root <folder the CSV paths start from>]

Or on Kaggle / Colab with internet (downloads what you do not have):

    python prepare_kaggle_inputs.py /kaggle/working/veristamp-m3-inputs --download-model \
        --test-manifest split_manifest.csv         # Rishitha's split file: the test photos come from COCO

Result:
    m3/best.pt  m3/test/*.jpg  code/VeriStamp/...  [m1/best.pt  m1/vslib_m1_final/]
    [genimage/splits/*.csv  genimage/images/...]  dataset-metadata.json  INPUTS.txt

Only the GenImage photos listed in the test split CSVs are copied, so the dataset stays small.
Running it again adds to the same folder: M1 and GenImage can come later.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DRIVE_FILE_ID = '1oB5NEaWLoLlbarJQnbLhmrG4znF0ZaEi'      # link in ../m3_watermark/README.md
COCO_VAL_URL = 'http://images.cocodataset.org/val2017/{}'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
SKIP_IN_CODE = shutil.ignore_patterns('.git', '__pycache__', '.ipynb_checkpoints', '.pytest_cache',
                                      'results', 'experiments', '*.pt', '*.pth', '*.ckpt', '*.zip')


def say(msg):
    print(msg, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------ M3 model
def _best_from_zip(data: bytes, name: str, depth=0):
    """Find M3's final best.pt in a zip, also inside zips within it (the Drive download holds one
    results zip per notebook; the final model is the one in results_3_final_model.zip)."""
    found = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for m in zf.namelist():
            if m.endswith('/best.pt') or m == 'best.pt':
                found.append((f'{name}/{m}', zf.read(m)))
            elif m.lower().endswith('.zip') and depth < 2:
                found += _best_from_zip(zf.read(m), f'{name}/{m}', depth + 1)
    return found


def get_m3_model(source: Path, target: Path):
    if source.suffix == '.pt':
        shutil.copy2(source, target)
        return str(source)
    candidates = _best_from_zip(source.read_bytes(), source.name)
    final = [c for c in candidates if 'results_3' in c[0] or 'final' in c[0].lower()]
    pick = final if len(final) == 1 else (candidates if len(candidates) == 1 else None)
    if not pick:
        raise SystemExit('Could not tell which best.pt is the final model. Found:\n  '
                         + '\n  '.join(c[0] for c in candidates)
                         + '\nExtract best.pt from results_3_final_model.zip and pass it with --m3-model.')
    target.write_bytes(pick[0][1])
    return pick[0][0]


def download_m3(work: Path):
    try:
        import gdown
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
        import gdown
    out = work / 'm3_drive_download.zip'
    if not out.exists():
        say('Downloading the M3 results from Google Drive ...')
        if not gdown.download(id=DRIVE_FILE_ID, output=str(out), quiet=False):
            raise SystemExit('The Drive download failed. The file may not be shared with "anyone with the '
                             'link": download it yourself and pass it with --m3-model.')
    return out


def check_m3_model(path: Path):
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    missing = {'encoder', 'decoder'} - set(ck)
    if missing:
        raise SystemExit(f'{path} is not an M3 checkpoint (missing {missing}). M1 also has a best.pt: '
                         'use the one from results_3_final_model.zip.')
    return ck.get('epoch')


# ------------------------------------------------------------------------------------- test photos
def manifest_test_files(manifest: Path):
    with open(manifest, newline='') as f:
        rows = list(csv.DictReader(f))
    if rows and 'split' in rows[0]:
        rows = [r for r in rows if r['split'] == 'test']
    col = next((c for c in ('filename', 'file_name', 'image', 'name') if rows and c in rows[0]), None)
    if col is None:
        raise SystemExit(f'{manifest} has no filename column.')
    return sorted({Path(r[col]).name for r in rows})      # also works with M3's test_predictions.csv


def download_coco(names, folder: Path, threads=8):
    def fetch(name):
        target = folder / name
        if target.exists() and target.stat().st_size > 0:
            return name, None
        for attempt in range(4):
            try:
                req = urllib.request.Request(COCO_VAL_URL.format(name),
                                             headers={'User-Agent': 'VeriStamp-M3-Research/1.0'})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                target.write_bytes(data)
                return name, None
            except Exception as e:                      # retry network errors
                err = e
                time.sleep(2 * (attempt + 1))
        return name, err
    with cf.ThreadPoolExecutor(threads) as ex:
        failed = [(n, e) for n, e in ex.map(fetch, names) if e]
    if failed:
        raise SystemExit(f'{len(failed)} COCO downloads failed, e.g. {failed[0]}. Run again to retry.')


def copy_test_dir(source: Path, folder: Path):
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    subdirs = [p for p in source.iterdir() if p.is_dir()]
    if not files and len(subdirs) == 1:
        return copy_test_dir(subdirs[0], folder)
    for p in files:
        shutil.copy2(p, folder / p.name)
    return len(files)


# ------------------------------------------------------------------------------------------ GenImage
def copy_genimage(splits: Path, root: Path | None, out: Path):
    """Copy the test split CSVs with image_path made relative (M1's CSVs hold absolute Colab paths),
    and, if root is given, only the photos they list."""
    sys.path.insert(0, str(HERE))
    from m3_eval import genimage_relpath
    (out / 'splits').mkdir(parents=True, exist_ok=True)
    n, missing = 0, []
    for name in ('test_seen.csv', 'test_unseen.csv'):
        src = splits / name
        if not src.exists():
            say(f'  {name} not found in {splits}, skipped')
            continue
        with open(src, newline='') as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            row['image_path'] = genimage_relpath(row)
        with open(out / 'splits' / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        if root is None:
            continue
        for row in rows:
            rel = Path(row['image_path'])
            target = out / 'images' / rel
            if target.exists():
                n += 1
                continue
            if not (root / rel).exists():
                missing.append(str(rel))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, target)
            n += 1
    if missing:
        say(f'  WARNING: {len(missing)} photos in the CSVs were not found under {root}, e.g. {missing[0]}')
    return n


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', type=Path, help='folder to build, e.g. out/veristamp-m3-inputs')
    ap.add_argument('--m3-model', type=Path, help='best.pt, results_3_final_model.zip or the whole Drive zip')
    ap.add_argument('--download-model', action='store_true', help='download the M3 results from Google Drive')
    ap.add_argument('--test-dir', type=Path, help='folder with the 500 test photos (hidden_data/test)')
    ap.add_argument('--test-manifest', type=Path, help='list of the test photos (filename column), e.g. test_predictions.csv '
                    'from results_3_final_model.zip. The notebooks download them from COCO on first run.')
    ap.add_argument('--download-photos', action='store_true', help='download the listed test photos now instead')
    ap.add_argument('--m1-checkpoint', type=Path)
    ap.add_argument('--m1-vslib', type=Path, help="the folder of .py files M1's notebook wrote (vslib_m1_final)")
    ap.add_argument('--genimage-splits', type=Path, help='folder with test_seen.csv and test_unseen.csv')
    ap.add_argument('--genimage-root', type=Path, help='the folder the image_path values start from')
    ap.add_argument('--no-code', action='store_true', help='leave out code/ (the notebooks then clone the repo)')
    ap.add_argument('--kaggle-user', default='YOUR_KAGGLE_USERNAME', help='for dataset-metadata.json')
    ap.add_argument('--n-test', type=int, default=500)
    ap.add_argument('--zip', action='store_true', help='also write <out>.zip to upload')
    a = ap.parse_args(argv)

    out = a.out.resolve()
    (out / 'm3' / 'test').mkdir(parents=True, exist_ok=True)
    info = {'built': time.strftime('%Y-%m-%d %H:%M:%S')}

    # M3 model
    model = out / 'm3' / 'best.pt'
    source = a.m3_model or (download_m3(out.parent) if a.download_model else None)
    if source:
        info['m3_model_from'] = get_m3_model(Path(source), model)
    if model.exists():
        try:
            info['m3_epoch'] = check_m3_model(model)
        except SystemExit:
            model.unlink()               # never leave a wrong checkpoint in the dataset
            raise
        info['m3_sha256'] = sha256(model)
        say(f"M3 model: epoch {info['m3_epoch']}, sha256 {info['m3_sha256'][:12]}...")

    # test photos
    test = out / 'm3' / 'test'
    if a.test_dir:
        say(f'Copied {copy_test_dir(a.test_dir, test)} test photos')
    manifest = out / 'm3' / 'test_manifest.csv'
    if a.test_manifest:
        names = manifest_test_files(a.test_manifest)
        with open(manifest, 'w', newline='') as f:
            csv.writer(f).writerows([['filename'], *[[n] for n in names]])
        say(f'Test photo list: {len(names)} COCO val2017 photos')
        if a.download_photos:
            say('Downloading them from COCO ...')
            download_coco(names, test)
    n_test = sum(1 for p in test.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    n_listed = len(manifest_test_files(manifest)) if manifest.exists() else 0
    info.update({'m3_test_photos': n_test, 'm3_test_photos_listed': n_listed})

    # M1 and GenImage (optional, can be added later)
    if a.m1_checkpoint:
        (out / 'm1').mkdir(exist_ok=True)
        shutil.copy2(a.m1_checkpoint, out / 'm1' / 'best.pt')
    if a.m1_vslib:
        shutil.copytree(a.m1_vslib, out / 'm1' / 'vslib_m1_final', dirs_exist_ok=True, ignore=SKIP_IN_CODE)
    if a.genimage_splits:
        n = copy_genimage(a.genimage_splits, a.genimage_root, out / 'genimage')
        say(f'GenImage: split CSVs copied, {n} test photos' + ('' if a.genimage_root else
            ' (add --genimage-root with the GenImage_native_n500 folder to copy the photos)'))

    # code
    if not a.no_code:
        code = out / 'code' / 'VeriStamp'
        if code.exists():
            shutil.rmtree(code)
        shutil.copytree(REPO_ROOT, code, ignore=SKIP_IN_CODE)

    # Kaggle metadata, and a summary of what is in the dataset
    meta = out / 'dataset-metadata.json'
    if not meta.exists():
        meta.write_text(json.dumps({'title': 'veristamp-m3-inputs', 'id': f'{a.kaggle_user}/veristamp-m3-inputs',
                                    'licenses': [{'name': 'other'}]}, indent=2))
    status = {
        'm3/best.pt (M3 model)': model.exists(),
        (f'm3/test ({n_test} photos)' if n_test else f'm3/test_manifest.csv ({n_listed} photos listed; downloaded '
         'from COCO on first run, needs Internet on)'): n_test == a.n_test or n_listed == a.n_test,
        'code/VeriStamp' + (' (left out: the notebooks clone the repo)' if a.no_code else ''):
            a.no_code or (out / 'code' / 'VeriStamp' / 'm3_experiments' / 'settings.py').exists(),
        'm1/best.pt (for notebooks 11, 12)': (out / 'm1' / 'best.pt').exists(),
        'm1/vslib_m1_final (for 11, 12)': (out / 'm1' / 'vslib_m1_final' / 'vs_models.py').exists(),
        'genimage/splits (for 10 to 12)': (out / 'genimage' / 'splits' / 'test_unseen.csv').exists(),
        'genimage/images (for 10 to 12)': (out / 'genimage' / 'images').exists(),
    }
    lines = [f"[{'x' if ok else ' '}] {k}" for k, ok in status.items()]
    (out / 'INPUTS.txt').write_text('veristamp-m3-inputs\n\n' + '\n'.join(lines) + '\n\n'
                                    + json.dumps(info, indent=2, default=str) + '\n')
    say('\n'.join(lines))
    ready = all(list(status.values())[:3])
    say('\nReady for notebooks 5 to 9.' if ready else '\nNOT READY for notebooks 5 to 9: see the unticked lines.')

    if a.zip:
        archive = shutil.make_archive(str(out), 'zip', out)
        say(f'Upload this on Kaggle (Datasets > New dataset, private): {archive} '
            f'({Path(archive).stat().st_size / 1e6:.0f} MB)')
    return 0 if ready else 1


if __name__ == '__main__':
    sys.exit(main())
