"""Export recorded notebook evidence using only the Python standard library."""
import base64
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / 'Convolutional Autoencoder.ipynb').read_text())
assets = ROOT / 'assets'
assets.mkdir(exist_ok=True)
images = []
losses = []
for cell in notebook['cells']:
    for output in cell.get('outputs', []):
        losses.extend((int(e), float(v)) for e, v in re.findall(
            r'Loss for epoch (\d+) = ([0-9.eE+-]+)', ''.join(output.get('text', []))))
        if 'image/png' in output.get('data', {}):
            images.append(base64.b64decode(''.join(output['data']['image/png'])))
names = ['training-original.png', 'reconstruction-01.png', 'reconstruction-02.png', 'reconstruction-03.png']
if len(images) != len(names) or len(losses) != 50:
    raise ValueError('Expected four saved figures and 50 recorded losses; inspect the notebook before exporting.')
for name, data in zip(names, images):
    (assets / name).write_bytes(data)
with (assets / 'training-loss.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'last_batch_mae'])
    writer.writerows(losses)
points = ' '.join(f'{80+(e-1)*920/49:.2f},{340-v/0.25*250:.2f}' for e,v in losses)
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="440" viewBox="0 0 1080 440" role="img" aria-labelledby="title desc">',
       '<title id="title">Recorded reconstruction loss</title><desc id="desc">Last training batch MAE for 50 epochs. This is not an epoch average or test score.</desc>',
       '<rect width="1080" height="440" rx="18" fill="#101b2b"/>',
       '<g font-family="Arial, sans-serif" fill="#e4edf8"><text x="50" y="46" font-size="24" font-weight="bold">Learning to reconstruct</text>',
       '<text x="50" y="73" font-size="14" fill="#a9b9cc">Recorded last-batch MAE · 50 epochs · lower is better</text>']
for value in [0,.05,.1,.15,.2,.25]:
    y=340-value/.25*250
    svg.append(f'<path d="M80 {y} H1000" stroke="#2a3a50"/><text x="37" y="{y+5}" font-size="12">{value:.2f}</text>')
svg.append(f'<polyline points="{points}" fill="none" stroke="#58dbc0" stroke-width="3"/>')
for epoch in [1,10,20,30,40,50]:
    x=80+(epoch-1)*920/49
    svg.append(f'<text x="{x}" y="365" text-anchor="middle" font-size="12">{epoch}</text>')
svg.append('<text x="540" y="392" text-anchor="middle" font-size="13">Epoch</text><text x="50" y="422" font-size="12" fill="#a9b9cc">Source: saved notebook logs. Shuffled final batches contain 3 images; fluctuations do not measure validation performance.</text></g></svg>')
(assets / 'training-loss.svg').write_text('\n'.join(svg))
print(f'Exported {len(images)} original figures, {len(losses)} loss records, and an SVG chart.')
print(f'First: {losses[0]}; minimum: {min(losses,key=lambda x:x[1])}; final: {losses[-1]}')
