"""Known-label DejaVu replay matching the measured ALPR recipe.

No country grammar or evaluation data. FreeFont families are not used. Rendering
is reproducible for a fixed Pillow/font version; the manifest records font hashes.
"""
import hashlib
import json
import random
import string
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_NAMES = ('DejaVuSans-Bold.ttf', 'DejaVuSansMono-Bold.ttf',
              'DejaVuSerif-Bold.ttf', 'DejaVuSansCondensed-Bold.ttf')


def _fonts(folder=None):
    paths = []
    for name in FONT_NAMES:
        try:
            font = ImageFont.truetype(str(Path(folder) / name) if folder else name, 16)
            paths.append(Path(font.path))
        except OSError:
            # Match the measured generator's per-font Pillow-default fallback.
            # Missing fonts are recorded in the replay specification below.
            paths.append(Path(folder) / name if folder else Path(name))
    return paths


def sample_plate(rng, fonts):
    text = ''.join(rng.choices(string.ascii_uppercase + string.digits, k=rng.randint(5, 12)))
    two = rng.random() < .55
    width, height = rng.choice([(180, 130), (200, 140), (170, 120)]) if two else rng.choice(
        [(280, 80), (260, 95), (240, 90)])
    dark = rng.random() < .25
    background = rng.choice([(22, 38, 30), (30, 30, 35), (42, 39, 25)]) if dark else rng.choice(
        [(244, 244, 231), (234, 218, 69), (240, 237, 220)])
    ink = (237, 237, 225) if dark else (rng.randint(0, 30),) * 3
    image = Image.new('RGB', (width, height), background)
    draw = ImageDraw.Draw(image)
    font_path = rng.choice(fonts)
    if two:
        split = rng.randint(2, min(5, len(text) - 2))
        lines = [text[:split], text[split:]]
    else:
        lines = [text]
    for i, line in enumerate(lines):
        size = int(height * (.45 if two else .65))
        def get_font(size):
            try:
                return ImageFont.truetype(str(font_path), size)
            except OSError:
                return ImageFont.load_default(size=size)
        font = get_font(size)
        while draw.textbbox((0, 0), line, font=font)[2] > width * .90 and size > 10:
            size -= 1
            font = get_font(size)
        box = draw.textbbox((0, 0), line, font=font)
        x = (width - box[2]) // 2
        y = (i + .5) * height / len(lines) - (box[3] - box[1]) / 2 - box[1]
        draw.text((x, y), line, font=font, fill=ink)
    if rng.random() < .5:
        draw.rectangle((1, 1, width - 2, height - 2), outline=ink, width=1)
    if rng.random() < .4:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.1, .7)))
    if rng.random() < .7:
        image = image.rotate(rng.uniform(-4, 4), Image.Resampling.BICUBIC,
                             fillcolor=background)
    return image, text


def generate_replay(folder, count=2048, seed=20261114, font_dir=None):
    folder = Path(folder)
    fonts = _fonts(font_dir)
    specification = {'count': int(count), 'seed': int(seed), 'Pillow': Image.__version__,
                     'font_sha256': {p.name: (hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file()
                         else 'missing: Pillow.load_default(size) fallback') for p in fonts},
                     'method': 'Random5..12 uppercase/digits;55%two-line;no country grammar'}
    manifest = folder / 'manifest.json'
    if manifest.exists():
        cached = json.loads(manifest.read_text())
        if cached['generator'] != specification:
            raise ValueError('Synthetic replay specification changed; choose a new output_dir')
        records = cached['rows']
        if not all((folder / row['path']).is_file() for row in records):
            raise FileNotFoundError('Synthetic replay manifest has missing images')
    else:
        folder.mkdir(parents=True, exist_ok=True)
        rng, records = random.Random(seed), []
        for i in range(count):
            image, text = sample_plate(rng, fonts)
            name = f'{i:05}.png'
            image.save(folder / name)
            records.append({'image': f'synthetic_plate/{name}', 'path': name,
                            'label': text, 'synthetic': True})
        manifest.write_text(json.dumps({'generator': specification, 'rows': records}, indent=2))
    return [{**r, 'path': str((folder / r['path']).resolve())} for r in records], specification
