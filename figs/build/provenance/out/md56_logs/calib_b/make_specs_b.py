"""Supp Fig 8 "b" rerun -- STAGE 2: one spec.json per recording, pointing BOTH tools at the
same bytes (the inputs/ tree built by transcode_b.sh), with each video's sha256.

    python3 make_specs_b.py <bench_b dir>

Writes <bench_b>/work/{slap_<sid>,md10_<sid>,calib18}/spec.json in the layout
figs/fig7_calib_score.py / _heldout.py / _equal_frames.py / _reference_cam.py already read
(cameras, board, size), plus `session_dir` (what calibrat3 is handed), `date`, `note`,
`tuning` and `sha256`, which fig7_calib_verify.py-style checks and deposit_b.py use.
"""
import sys, os, json, glob, re, csv

bench = sys.argv[1]
IN = os.path.join(bench, 'inputs')
W = os.path.join(bench, 'work')
BOARD = dict(board_x=8, board_y=11, square_length=24.0, marker_length=18.75, marker_bits=4, dict_size=1000)
sha = {}
with open(os.path.join(bench, 'logs', 'transcode_provenance.tsv')) as fh:
    for r in csv.DictReader(fh, delimiter='\t'):
        sha[r['output']] = dict(output_sha256=r['output_sha256'], source=r['source'],
                                source_sha256=r['source_sha256'], mode=r['mode'])

TUNING = {'slap_10072022120554', 'calib18'}          # used to choose calibrat3's defaults
DATES = {'10072022120554': '2022-10-07', '10192022174304': '2022-10-19',
         '10212022155401': '2022-10-21', '10302022010116': '2022-10-30'}


def write(key, cams, size, session_dir, date, note):
    d = os.path.join(W, key); os.makedirs(d, exist_ok=True)
    spec = dict(cameras=cams, board=BOARD, size=size, session_dir=session_dir, date=date,
                note=note, tuning=key in TUNING, sha256={c: sha.get(v) for c, v in cams})
    missing = [c for c, v in cams if not sha.get(v)]
    if missing:
        raise SystemExit(f'{key}: no provenance row for {missing} -- run transcode_b.sh first')
    json.dump(spec, open(os.path.join(d, 'spec.json'), 'w'), indent=1)
    print(f'{key}: {len(cams)} cameras, {size}, tuning={key in TUNING}')


for sid, date in DATES.items():
    sd = os.path.join(IN, 'slap8', sid)
    cams = [[os.path.basename(v).split('-')[1], v] for v in sorted(glob.glob(f'{sd}/*.mp4'))]
    write(f'slap_{sid}', cams, [1280, 1024], sd, date,
          'TUNING RECORDING (same bytes as cal_test2)' if f'slap_{sid}' in TUNING else 'held out')

for sd in sorted(glob.glob(os.path.join(IN, 'md10', '*'))):
    sid = os.path.basename(sd)
    cams = [[os.path.basename(v).split('-')[0], v] for v in sorted(glob.glob(f'{sd}/*-calibration.mp4'))]
    write(f'md10_{sid}', cams, [1280, 1024], sd, f'{sid[:4]}-{sid[4:6]}-{sid[6:8]}', 'held out')

c18 = os.path.join(IN, 'calib18')
cams = [[os.path.basename(d), f'{d}/calibration_images/calib.mp4']
        for d in sorted(glob.glob(f'{c18}/Camera*'), key=lambda p: int(re.search(r'(\d+)$', p).group(1)))]
write('calib18', cams, [1680, 1200], c18, '2024-12-06', 'tuning recording, single session')
