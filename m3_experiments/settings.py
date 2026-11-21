"""Paths and fixed settings for the M3 experiment notebooks (5 to 12).

On Kaggle nothing needs editing: the paths are found in the attached input dataset, laid out as in
README.md ("Running on Kaggle"). Elsewhere (Colab, lab PC, laptop), edit the paths below once.
Every notebook imports this file, so all notebooks use the same model, test photos and seeds.
Each path can also be set with an environment variable of the same name, for example M3E_MODEL_PATH.

Nothing here changes the M3 files in ../m3_watermark. m3_watermark.py is imported from that folder
as it is, and best.pt is only read, never written.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


ON_KAGGLE = Path('/kaggle/input').is_dir()
KAGGLE_INPUT = Path('/kaggle/input')


def _kaggle(rel):
    """Find <rel> (for example 'm3/best.pt') up to three folders deep in the Kaggle inputs, so the
    dataset can have any name and Kaggle's extra folder from an uploaded zip does not matter."""
    for depth in range(4):
        hits = sorted(KAGGLE_INPUT.glob('/'.join(['*'] * depth + [rel])))
        if hits:
            return hits[0]
    return KAGGLE_INPUT / 'veristamp-m3-inputs' / rel       # shown in the error message if missing


def _path(name, default, kaggle_rel=None):
    if name in os.environ:
        return Path(os.environ[name]).expanduser()
    if ON_KAGGLE and kaggle_rel:
        return _kaggle(kaggle_rel) if not kaggle_rel.startswith('/') else Path(kaggle_rel)
    return Path(default).expanduser()


# ---- Edit these (not needed on Kaggle) -----------------------------------------------------------------------------------
# Folder that holds m3_watermark.py. The repo copy is used as it is.
M3_DIR = _path('M3E_M3_DIR', REPO_ROOT / 'm3_watermark')

# The final model: best.pt from results_3_final_model.zip (Drive link in ../m3_watermark/README.md).
MODEL_PATH = _path('M3E_MODEL_PATH', '/content/drive/MyDrive/VeriStamp_M3/best.pt', 'm3/best.pt')

# The 500 COCO test photos (hidden_data/test from Rishitha). Used for testing only, never training.
TEST_DIR = _path('M3E_TEST_DIR', '/content/drive/MyDrive/VeriStamp_M3/hidden_data/test', 'm3/test')

# Optional list of the test photos (filename column). If TEST_DIR has no photos, setup() downloads
# them from COCO val2017 into WORK_DIR/test_photos (needs internet; on Kaggle turn Internet on).
TEST_MANIFEST = _path('M3E_TEST_MANIFEST', HERE / 'm3_test_manifest.csv', 'm3/test_manifest.csv')

# Large working files: stamped and edited photos. Keep this OUTSIDE the git repo.
WORK_DIR = _path('M3E_WORK_DIR', '/content/m3_work', '/kaggle/working/m3_work')

# Small result tables (CSV and Markdown). These are committed to git.
RESULTS_DIR = _path('M3E_RESULTS_DIR', HERE / 'results')

# GenImage (tasks 6, 9, 10): the folder the M1 split paths are relative to, and the split folder
# with test_seen.csv and test_unseen.csv. Ask Kruthika or Supriya for the exact locations.
GENIMAGE_ROOT = _path('M3E_GENIMAGE_ROOT', '/content/drive/MyDrive/Datasets/GenImage_native_n500',
                      'genimage/images')
GENIMAGE_SPLIT_DIR = _path(
    'M3E_GENIMAGE_SPLIT_DIR',
    '/content/drive/MyDrive/VeriStamp/data/splits/'
    'genimage_seen-ADM-VQDM_unseen-stable_diffusion_v_1_4_min224_n500_seed42__m1_final_n500',
    'genimage/splits')

# M1 (task 9, 10): the Python modules the M1 notebook wrote, and M1's best.pt.
M1_VSLIB_DIR = _path('M3E_M1_VSLIB_DIR', '/content/drive/MyDrive/VeriStamp/vslib_m1_final',
                     'm1/vslib_m1_final')
M1_CHECKPOINT = _path('M3E_M1_CHECKPOINT',
                      '/content/drive/MyDrive/VeriStamp/experiments/m1_final_n500/hp_fft/best.pt',
                      'm1/best.pt')
# ---------------------------------------------------------------------------------------------------

# Fixed settings. Do not change them between runs, or results stop being comparable.
SEED = 42
N_TEST_PHOTOS = int(os.environ.get('M3E_N_TEST_PHOTOS', 500))   # the guide's 500 test photos
VAE_NAME = 'stabilityai/sd-vae-ft-mse'
INPAINT_NAME = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
INPAINT_PROMPT = 'a natural photo'

# Placeholder Uncertain thresholds from configs/base.yaml. Replace with the fitted values from
# checkpoints/thresholds_<model>.json once the team has chosen them (docs/decisions.md, item 10).
THRESHOLDS = {'m1_resnet': (0.35, 0.65), 'm2_clip': (0.35, 0.65), 'fusion': (0.35, 0.65),
              'dummy': (0.35, 0.65)}


def _photos_in(folder):
    if not folder.is_dir():
        return 0
    exts = {'.jpg', '.jpeg', '.png'}
    n = sum(1 for p in folder.iterdir() if p.suffix.lower() in exts)
    subdirs = [p for p in folder.iterdir() if p.is_dir()]
    return n if n or len(subdirs) != 1 else _photos_in(subdirs[0])


def setup():
    """Put m3_watermark.py and the repo on the import path, create the output folders, get the test
    photos if needed, and check that everything exists. Returns the m3_watermark module."""
    global TEST_DIR
    for p in (M3_DIR, REPO_ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if _photos_in(TEST_DIR) < N_TEST_PHOTOS and TEST_MANIFEST.exists():
        from prepare_kaggle_inputs import download_coco, manifest_test_files
        names = manifest_test_files(TEST_MANIFEST)[:N_TEST_PHOTOS] if N_TEST_PHOTOS < 500 else \
            manifest_test_files(TEST_MANIFEST)
        TEST_DIR = WORK_DIR / 'test_photos'
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        if _photos_in(TEST_DIR) < len(names):
            print(f'Downloading {len(names)} test photos from COCO val2017 into {TEST_DIR} ...', flush=True)
            download_coco(names, TEST_DIR)
    missing = [str(p) for p in (M3_DIR / 'm3_watermark.py', MODEL_PATH, TEST_DIR) if not p.exists()]
    if missing:
        hint = ('On Kaggle: attach the input dataset (README.md, "Running on Kaggle"). Found under '
                '/kaggle/input:\n' + list_kaggle_inputs()) if ON_KAGGLE else \
               'Set the paths at the top of m3_experiments/settings.py.'
        raise FileNotFoundError('Missing: ' + ', '.join(missing) + '\n' + hint)
    import m3_watermark
    return m3_watermark


def list_kaggle_inputs(max_depth=3, max_lines=60):
    """What is attached under /kaggle/input, three folders deep, to see where files ended up."""
    lines = []
    def walk(p, depth):
        for c in sorted(p.iterdir())[:15]:
            if len(lines) >= max_lines:
                return
            n = sum(1 for _ in c.iterdir()) if c.is_dir() else None
            lines.append('  ' * depth + c.name + (f'/  ({n} items)' if c.is_dir() else ''))
            if c.is_dir() and depth + 1 < max_depth:
                walk(c, depth + 1)
    if KAGGLE_INPUT.is_dir():
        walk(KAGGLE_INPUT, 0)
    return '\n'.join(lines) or '  (nothing attached)'
