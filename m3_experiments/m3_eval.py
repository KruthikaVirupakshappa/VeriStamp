"""Shared code for the M3 experiment notebooks: payloads, stamping, edits, scoring and tables.

It only USES m3_watermark.py (load_m3, embed, read_bits, encode_payload, decode_payload) and never
changes it. Every notebook stamps the same photos with the same payloads, so results line up.

Words used in the tables (the same as the handover guide):
  bit accuracy          share of the 32 bits read correctly; 50% means guessing
  valid verdict         share of photos whose message passed the checksum AND gave the right verdict
  invalidation rate     share of photos whose stamp is no longer valid (checksum fails)
  wrong but valid       checksum passed but the message differs from what was stamped (should be ~0)
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

try:
    from tqdm.auto import tqdm
except ImportError:                     # tqdm is optional
    def tqdm(x, **_):
        return x

VERDICTS = ('Real', 'AI', 'Uncertain')
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


# ---------------------------------------------------------------- files and fixed randomness
def list_photos(folder, n=None):
    """The photos in folder, sorted by name. If the folder only holds one subfolder (as when an
    uploaded zip had a top folder), the photos in that subfolder."""
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    subdirs = [p for p in folder.iterdir() if p.is_dir()]
    if not files and len(subdirs) == 1:
        return list_photos(subdirs[0], n)
    return files[:n] if n else files


def photo_seed(name, salt='', seed=42):
    """A fixed integer seed for one photo and one purpose. Same photo, same seed, on every run."""
    return zlib.crc32(f'{seed}:{salt}:{name}'.encode())


def payload_for(name, seed=42):
    """The fixed verdict, P(AI) and image tag for one photo. Used by every notebook and every edit."""
    rng = np.random.default_rng(photo_seed(name, 'payload', seed))
    return {'verdict': VERDICTS[int(rng.integers(3))], 'p_ai': round(float(rng.random()), 4),
            'image_tag': int(rng.integers(2 ** 14))}


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def open_rgb(path):
    with Image.open(path) as im:
        return im.convert('RGB')


def save_as(image, fmt='PNG', quality=None):
    """Save to memory and open again, exactly like saving a file to disk and reloading it."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **({'quality': int(quality)} if quality else {}))
    buffer.seek(0)
    with Image.open(buffer) as im:
        return im.convert('RGB')


# ---------------------------------------------------------------- stamping the test set once
def stamp_test_set(mw, m3, files, out_dir, model_path, seed=42):
    """Stamp every photo with its fixed payload and save it as PNG in out_dir/stamped.

    Runs once; later notebooks reuse the files. A small JSON records which checkpoint made them,
    so stamps from a different model are never mixed in by accident.
    Returns a DataFrame with one row per photo.
    """
    out_dir = Path(out_dir)
    stamped_dir = out_dir / 'stamped'
    stamped_dir.mkdir(parents=True, exist_ok=True)
    info_file = out_dir / 'stamped_info.json'
    sha = file_sha256(model_path)
    if info_file.exists():
        info = json.loads(info_file.read_text())
        if info.get('model_sha256') != sha or info.get('seed') != seed:
            raise RuntimeError(f'{stamped_dir} was made with another model or seed. '
                               'Use a new WORK_DIR or delete that folder.')
    rows = []
    for path in tqdm(files, desc='Stamping'):
        p = payload_for(path.name, seed)
        target = stamped_dir / (path.stem + '.png')
        if not target.exists():
            stamped = mw.embed(m3, open_rgb(path), p['verdict'], p['p_ai'], image_tag=p['image_tag'])
            stamped.save(target)
        expected = mw.encode_payload(p['verdict'], p['p_ai'], image_tag=p['image_tag'])
        with Image.open(path) as im:
            w, h = im.size
        rows.append({'name': path.name, 'original': str(path), 'stamped': str(target), 'width': w,
                     'height': h, **p, 'expected_bits': ''.join(map(str, expected))})
    info_file.write_text(json.dumps({'model_path': str(model_path), 'model_sha256': sha,
                                     'model_epoch': m3.epoch, 'seed': seed, 'n_photos': len(rows),
                                     'created': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=2))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'stamped_payloads.csv', index=False)
    return df


# ---------------------------------------------------------------- reading a stamp back
def score_read(mw, m3, image, expected_bits, verdict):
    """Read the stamp from a PIL image and compare it with what was stamped."""
    bits = mw.read_bits(m3, image)
    expected = [int(c) for c in expected_bits]
    decoded = mw.decode_payload(bits)
    exact = bits == expected
    return {'bit_accuracy': float(np.mean([a == b for a, b in zip(bits, expected)])),
            'valid': bool(decoded['valid']),
            'valid_verdict': bool(decoded['valid'] and decoded['verdict'] == verdict),
            'all_bits_correct': bool(exact),
            'wrong_but_valid': bool(decoded['valid'] and not exact),
            'read_verdict': decoded['verdict'], 'read_bits': ''.join(map(str, bits))}


def run_edit(mw, m3, records, edit, edit_name, source='stamped', save_dir=None):
    """Apply edit(image, row) to every photo, read the stamp back, return one row per photo.

    source='stamped' edits the stamped PNG; source='original' edits the unstamped photo, to count
    photos without a stamp that are wrongly read as valid.
    """
    out = []
    for _, r in tqdm(records.iterrows(), total=len(records), desc=edit_name, leave=False):
        image = open_rgb(r[source])
        edited = edit(image, r)
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            edited.save(Path(save_dir) / (Path(r['name']).stem + '.png'))
        res = score_read(mw, m3, edited, r['expected_bits'], r['verdict'])
        out.append({'edit': edit_name, 'source': source, 'name': r['name'], **res})
    return pd.DataFrame(out)


def summarise(per_photo, by=('edit',)):
    """The guide's table: bit accuracy, valid verdict and invalidation rate per edit."""
    g = per_photo.groupby(list(by), sort=False)
    table = pd.DataFrame({
        'n_photos': g.size(),
        'bit_accuracy': g['bit_accuracy'].mean(),
        'valid_verdict': g['valid_verdict'].mean(),
        'invalidation_rate': 1 - g['valid'].mean(),
        'all_32_bits_correct': g['all_bits_correct'].mean(),
        'wrong_but_valid': g['wrong_but_valid'].sum(),
    })
    return table.reset_index()


def false_valid_summary(per_photo):
    """For photos WITHOUT a stamp: how many are wrongly read as carrying a valid stamp."""
    g = per_photo.groupby('edit', sort=False)
    return pd.DataFrame({'n_photos': g.size(), 'read_as_valid': g['valid'].sum(),
                         'false_valid_rate': g['valid'].mean()}).reset_index()


def as_percent(table, cols=('bit_accuracy', 'valid_verdict', 'invalidation_rate',
                            'all_32_bits_correct', 'false_valid_rate')):
    t = table.copy()
    for c in cols:
        if c in t:
            t[c] = (100 * t[c]).map(lambda v: f'{v:.2f}%')
    return t


def to_markdown(df, floatfmt='.4f'):
    """A DataFrame as a Markdown table, without the optional tabulate package."""
    cols = list(df.columns)
    fmt = lambda v: format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines)


def save_results(results_dir, stem, per_photo=None, summary=None, meta=None):
    """Write <stem>_per_photo.csv, <stem>_summary.csv/.md and <stem>_meta.json to results_dir."""
    d = Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    if per_photo is not None:
        per_photo.to_csv(d / f'{stem}_per_photo.csv', index=False)
    if summary is not None:
        summary.to_csv(d / f'{stem}_summary.csv', index=False)
        (d / f'{stem}_summary.md').write_text(to_markdown(as_percent(summary)) + '\n')
    if meta is not None:
        (d / f'{stem}_meta.json').write_text(json.dumps(meta, indent=2, default=str))


def environment_info():
    import platform
    import torch
    info = {'python': platform.python_version(), 'torch': torch.__version__,
            'cuda': torch.cuda.is_available(), 'date': time.strftime('%Y-%m-%d %H:%M:%S')}
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
    for pkg in ('diffusers', 'transformers', 'lpips', 'skimage'):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            pass
    return info


# ---------------------------------------------------------------- task 2: screenshot simulation
def resize_by(image, scale, resample=Image.BILINEAR):
    w, h = image.size
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), resample)


def blur(image, radius=0.5):
    return image.filter(ImageFilter.GaussianBlur(radius))


def colour_shift(image, name, max_change=0.05, seed=42):
    """Brightness and contrast each changed by a fixed random factor between -5% and +5%."""
    rng = np.random.default_rng(photo_seed(name, 'colour', seed))
    b, c = rng.uniform(1 - max_change, 1 + max_change, size=2)
    return ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(b)).enhance(c)


def screenshot(image, name, scale=0.8, radius=0.5, max_change=0.05, seed=42):
    """The guide's sequence: resize to 80%, blur 0.5, brightness/contrast up to 5%, resize back, PNG."""
    size = image.size
    x = resize_by(image, scale)
    x = blur(x, radius)
    x = colour_shift(x, name, max_change, seed)
    x = x.resize(size, Image.BILINEAR)
    return save_as(x, 'PNG')


SCREENSHOT_EDITS = {
    'No edit (PNG)': lambda im, r: save_as(im, 'PNG'),
    'Resize to 80% only': lambda im, r: save_as(resize_by(im, 0.8), 'PNG'),
    'Resize to 80% and back': lambda im, r: save_as(resize_by(im, 0.8).resize(im.size, Image.BILINEAR), 'PNG'),
    'Gaussian blur r=0.5 only': lambda im, r: save_as(blur(im, 0.5), 'PNG'),
    'Brightness/contrast +-5% only': lambda im, r: save_as(colour_shift(im, r['name']), 'PNG'),
    'Screenshot simulation (all steps)': lambda im, r: screenshot(im, r['name']),
    'Screenshot simulation + JPEG 90': lambda im, r: save_as(screenshot(im, r['name']), 'JPEG', 90),
}


# ---------------------------------------------------------------- task 3: image quality
def psnr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(np.asarray(a), np.asarray(b), channel_axis=2, data_range=255))


def load_lpips(device):
    import lpips
    return lpips.LPIPS(net='alex', verbose=False).to(device).eval()


def lpips_distance(model, a, b, device):
    import torch
    to = lambda im: (torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1)
                     .permute(2, 0, 1).unsqueeze(0).to(device))
    with torch.no_grad():
        return float(model(to(a), to(b)).item())


# ---------------------------------------------------------------- task 4: VAE round trip
def load_vae(device, name='stabilityai/sd-vae-ft-mse'):
    import torch
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()


def vae_roundtrip(vae, image, size=512):
    """Resize to size x size, encode to the latent space, decode, resize back, save as PNG.
    Uses the mean of the latent distribution, so the result is the same on every run."""
    import torch
    w, h = image.size
    device = next(vae.parameters()).device
    x = torch.from_numpy(np.asarray(image.resize((size, size), Image.BICUBIC), dtype=np.float32) / 127.5 - 1)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
        y = vae.decode(z).sample
    y = ((y[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return save_as(Image.fromarray(y).resize((w, h), Image.BICUBIC), 'PNG')


def resize_roundtrip(image, size=512):
    """Control for the VAE test: the same resizing, without the VAE."""
    w, h = image.size
    return save_as(image.resize((size, size), Image.BICUBIC).resize((w, h), Image.BICUBIC), 'PNG')


# ---------------------------------------------------------------- task 5: inpainting
def load_inpainter(device, name='stable-diffusion-v1-5/stable-diffusion-inpainting'):
    import torch
    from diffusers import AutoPipelineForInpainting
    dtype = torch.float16 if str(device).startswith('cuda') else torch.float32
    # The safety filter can replace an image it wrongly flags with a black one, which would spoil the
    # measurement; these are COCO photos and a neutral prompt.
    pipe = AutoPipelineForInpainting.from_pretrained(name, torch_dtype=dtype, safety_checker=None,
                                                     requires_safety_checker=False).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def mask_box(width, height, fraction, name, seed=42):
    """A rectangle covering `fraction` of the photo, with a fixed random shape and position."""
    rng = np.random.default_rng(photo_seed(name, f'mask{fraction}', seed))
    area = fraction * width * height
    aspect = float(rng.uniform(0.75, 1.333))
    mw = int(round(min(width, np.sqrt(area * aspect))))
    mh = int(round(min(height, area / max(mw, 1))))
    mw = int(round(min(width, area / max(mh, 1))))           # keep the area when one side was clipped
    left = int(rng.integers(0, width - mw + 1))
    top = int(rng.integers(0, height - mh + 1))
    return left, top, left + mw, top + mh


def box_mask(width, height, box):
    mask = Image.new('L', (width, height), 0)
    ImageDraw.Draw(mask).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=255)
    return mask


def inpaint(pipe, image, box, prompt, seed, steps=30, guidance=7.5, size=512):
    """Redraw the box with the inpainting model.

    Returns (composited, raw). `composited` keeps every pixel outside the box from the input photo,
    as photo editors do; only the box is new. `raw` is the model's whole output, in which every pixel
    has passed through the Stable Diffusion VAE. The guide's test is the composited one; raw is
    reported as an extra row because many simple tools return it as it is.
    """
    import torch
    w, h = image.size
    mask = box_mask(w, h, box)
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    out = pipe(prompt=prompt, image=image.resize((size, size), Image.BICUBIC),
               mask_image=mask.resize((size, size), Image.NEAREST), height=size, width=size,
               num_inference_steps=steps, guidance_scale=guidance, generator=generator).images[0]
    raw = out.convert('RGB').resize((w, h), Image.BICUBIC)
    composited = image.copy()
    composited.paste(raw, (0, 0), mask)
    return save_as(composited, 'PNG'), save_as(raw, 'PNG')


# ---------------------------------------------------------------- GenImage split (tasks 6, 9, 10)
def genimage_relpath(row):
    """The image's path relative to the GenImage folder. M1's split CSVs store absolute Colab paths
    (/content/drive/MyDrive/Datasets/GenImage_native_n500/...) in image_path; the relative path is in
    image_id."""
    p, i = str(row['image_path']), str(row.get('image_id', ''))
    if Path(p).is_absolute():
        return i if i and p.replace('\\', '/').endswith(i) else Path(p).name
    return p


def genimage_path(row, root):
    p = Path(str(row['image_path']))
    return p if p.is_absolute() and p.exists() else Path(root) / genimage_relpath(row)


def load_genimage_test(split_dir, root, n_per_class=300, seed=42, splits=('test_seen', 'test_unseen'),
                       min_side=128):
    """Sample n_per_class Real and n_per_class AI photos from the detectors' test split CSVs.

    Expects M1's split files (columns image_id, image_path, label, generator). Paths that are not
    absolute are taken relative to `root`. Label 1 = AI, 0 = Real.
    """
    frames = []
    for s in splits:
        f = Path(split_dir) / f'{s}.csv'
        if f.exists():
            frames.append(pd.read_csv(f).assign(split=s))
    if not frames:
        raise FileNotFoundError(f'No test split CSVs in {split_dir}. Ask Kruthika or Supriya for them.')
    df = pd.concat(frames, ignore_index=True).drop_duplicates('image_id')
    df['path'] = [str(genimage_path(r, root)) for r in df.to_dict('records')]
    for col in ('width', 'height'):
        if col in df:
            df = df[df[col].fillna(min_side) >= min_side]
    parts = [g.sample(n=min(n_per_class, len(g)), random_state=seed) for _, g in df.groupby('label')]
    return pd.concat(parts).sort_values(['label', 'image_id']).reset_index(drop=True)


# ---------------------------------------------------------------- RQ3 summary (task 9)
def to_verdict(p_ai, t_low, t_high):
    """Same rule as models/thresholds.py."""
    return 'Real' if p_ai < t_low else ('AI' if p_ai > t_high else 'Uncertain')


def bootstrap_ci(values, n=2000, seed=42):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rq3_summary(df, thresholds):
    """df: one row per photo and detector with columns detector, label (1 = AI), p_before, p_after.
    Returns the guide's RQ3 table plus the details behind it."""
    from sklearn.metrics import roc_auc_score
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    rows = []
    for det, g in df.groupby('detector', sort=False):
        t_low, t_high = thresholds.get(det, (0.35, 0.65))
        vb = g['p_before'].map(lambda p: to_verdict(p, t_low, t_high))
        va = g['p_after'].map(lambda p: to_verdict(p, t_low, t_high))
        d = g['p_after'] - g['p_before']
        y = g['label'].astype(int)
        row = {'detector': det, 'n_real': int((y == 0).sum()), 'n_ai': int((y == 1).sum()),
               'score_change_real': float(d[y == 0].mean()), 'score_change_ai': float(d[y == 1].mean())}
        row['ci95_real_low'], row['ci95_real_high'] = bootstrap_ci(d[y == 0])
        row['ci95_ai_low'], row['ci95_ai_high'] = bootstrap_ci(d[y == 1])
        row['mean_abs_change'] = float(d.abs().mean())
        row['max_abs_change'] = float(d.abs().max())
        if wilcoxon is not None and (d != 0).any():
            row['wilcoxon_p'] = float(wilcoxon(g['p_after'], g['p_before']).pvalue)
        flips = vb.values != va.values
        row.update({'verdict_flips': int(flips.sum()), 'flip_rate': float(flips.mean()),
                    'real_to_ai': int(((vb == 'Real') & (va == 'AI')).sum()),
                    'ai_to_real': int(((vb == 'AI') & (va == 'Real')).sum()),
                    'into_uncertain': int(((vb != 'Uncertain') & (va == 'Uncertain')).sum()),
                    'out_of_uncertain': int(((vb == 'Uncertain') & (va != 'Uncertain')).sum()),
                    'accuracy_before': float(((g['p_before'] >= 0.5).astype(int) == y).mean()),
                    'accuracy_after': float(((g['p_after'] >= 0.5).astype(int) == y).mean())})
        if y.nunique() == 2:
            row['auroc_before'] = float(roc_auc_score(y, g['p_before']))
            row['auroc_after'] = float(roc_auc_score(y, g['p_after']))
        rows.append(row)
    return pd.DataFrame(rows)
