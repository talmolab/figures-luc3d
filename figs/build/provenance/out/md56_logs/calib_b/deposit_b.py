"""Build figs/out/fig7_calibration_b.json from the "b" rerun (calib-bench-b/work).

Same schema the fig7 panels read (hist_edges, datasets{slap8,md10,calib18}{cameras, size,
n_cameras, recordings, detsets, timing}, calib18.ablation, synthetic), so the panels run
unmodified with FIG_VARIANT=b FIG_TAG=b. Differences from fig7_calib_deposit_unified.py:

  * EDGES/box_stats come from figs/fig7_calib_deposit.py (the unified builder imported them
    from a /tmp bench dir that no longer exists);
  * calib18's E/F arms are the END-TO-END calibrat3 run (the same run whose time is in H),
    scored together with Anipose only -- the published deposit used the ablation re-solve
    `default` and an ablation-intersected scoring set (644,539 obs) for E/F;
  * the Anipose arm is whatever anipose_arm.json chose by the PRE-REGISTERED rule in
    run_all_b.sh (end-to-end run unless it diverged), and the divergence count is deposited;
  * `timing` holds the MEDIAN of the repeated, same-cpuset, quiet-gated runs in the panel's
    keys, plus every rep, the cpuset, the load measured outside it and the CPU seconds;
  * every recording carries its controls block (equal frames / same reference /
    aniposelib-on-calibrat3-corners), scored on that recording's own shared set;
  * provenance: input sha256s, versions, git SHAs.

    python3 deposit_b.py <bench_b dir> <out.json>
"""
import sys, os, json, glob, subprocess, statistics, pickle, hashlib
import numpy as np

FIGS = '/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs'
sys.path.insert(0, FIGS)
from fig7_calib_deposit import EDGES, box_stats  # noqa: E402

bench, out_path = sys.argv[1], sys.argv[2]
W = os.path.join(bench, 'work')
assert not os.path.exists(out_path) or out_path.endswith('_b.json'), 'refusing to overwrite a non-b deposit'


def arm_record(errs_by_cam):
    a = dict(cameras={}, hist=np.zeros(len(EDGES) - 1))
    pooled = []
    for cam, e in errs_by_cam.items():
        a['cameras'][cam] = box_stats(e)
        a['hist'] += np.histogram(e, EDGES)[0]
        pooled.append(e)
    p = np.concatenate(pooled)
    a['overall'] = box_stats(p)
    a['overall']['p50'] = a['overall']['med']
    qs = np.concatenate([np.arange(0, 99, 1.0), np.linspace(99, 100, 21)])
    a['quantiles'] = dict(q=qs.tolist(), v=np.percentile(p, qs).tolist())
    a['hist'] = a['hist'].astype(int).tolist()
    return a


def load_npz(path, keep=None):
    z = np.load(path)
    arms = {}
    for key in z.files:
        lab, cam = key.split('|', 1)
        if keep is None or lab in keep:
            arms.setdefault(lab, {})[cam] = z[key]
    return arms


def jl(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def summarize_controls(d):
    s = jl(os.path.join(d, 'controls', 'scores.json'))
    if not s:
        return None
    return {det: {lab: v['overall'] for lab, v in arms.items() if lab != '_meta'} | {'_meta': arms.get('_meta')}
            for det, arms in s.items()}


def rows_digest(pkl):
    a = pickle.load(open(pkl, 'rb'))
    h = hashlib.sha256()
    for cam in a['rows']:
        for r in cam:
            h.update(np.asarray(r['framenum']).tobytes()); h.update(np.asarray(r['ids']).tobytes())
            h.update(np.asarray(r['corners'], dtype=np.float64).tobytes())
    return h.hexdigest()


def timing_block(d):
    """Median over reps of each tool's stage times, in the keys fig7_02_runtime.py reads."""
    tdir = os.path.join(d, 'timing')
    reps = sorted({int(p.rsplit('_r', 1)[1].split('.')[0]) for p in glob.glob(f'{tdir}/anipose_summary_r*.json')})
    if not reps:
        return None
    A = [jl(f'{tdir}/anipose_summary_r{r}.json') for r in reps]
    C = [jl(f'{tdir}/calibrat3_800_r{r}.timing.json') for r in reps]
    TA = [jl(f'{tdir}/timed_anipose_r{r}.json') for r in reps]
    TC = [jl(f'{tdir}/timed_calibrat3_800_r{r}.json') for r in reps]
    med = lambda xs: float(statistics.median(xs))
    c3 = dict(C[0]); c3['timing'] = {k: med([c['timing'][k] for c in C])
                                     for k in ('load_s', 'detection_s', 'intrinsics_s', 'extrinsics_s', 'sba_s', 'total_s')}
    ap = dict(A[0])
    for k in ('detection_s_wall', 'detection_s_cpu', 'calibration_s', 'calibration_s_cpu'):
        ap[k] = med([a[k] for a in A])
    strip = lambda t: {k: v for k, v in t.items() if k != 'samples'}
    out = {'calibrat3_800': c3, 'anipose': ap,
           'protocol': dict(reps=reps, statistic='median over reps',
                            order='rep odd: Anipose then calibrat3; rep even: calibrat3 then Anipose',
                            cpulist=TA[0]['cpulist'], n_cpus=TA[0]['n_cpus']),
           'reps': dict(anipose=[dict(summary=a, run=strip(t)) for a, t in zip(A, TA)],
                        calibrat3_800=[dict(summary=c, run=strip(t)) for c, t in zip(C, TC)])}
    allf = jl(f'{tdir}/calibrat3_all_r1.timing.json')
    if allf:
        out['calibrat3_all'] = allf
    out['max_outside_busy_cores'] = max(t['outside_busy_cores_max'] for t in TA + TC)
    out['all_quiet_gates_passed'] = all(t['quiet_gate_passed'] for t in TA + TC)
    # determinism checks the published figure asserted but this deposit now records
    out['anipose_detections_identical_across_reps'] = len({rows_digest(f'{tdir}/anipose_rows_r{r}.pkl') for r in reps
                                                          if os.path.exists(f'{tdir}/anipose_rows_r{r}.pkl')}) == 1
    out['calibrat3_toml_identical_across_reps'] = len({open(f'{tdir}/calibrat3_800_r{r}.toml', 'rb').read()
                                                      for r in reps}) == 1
    return out


def recording(d):
    spec = jl(os.path.join(d, 'spec.json'))
    arms = load_npz(os.path.join(d, 'scores_anipose.npz'), keep={'calibrat3_800', 'anipose'})
    sc = jl(os.path.join(d, 'scores.json'))
    c3t = jl(os.path.join(d, 'calibrat3_800.timing.json'), {})
    aps = jl(os.path.join(d, 'anipose_summary.json'), {})
    #: the frame budgets the caption/methods quote (867-1,003; 9,029; 12,621; 2,606/623 vs 882/378),
    #: now sourced per recording instead of from lost scratch files
    frames = dict(total_frames=c3t.get('totalFrames'), calibrat3_frames_sampled=c3t.get('detectionFrames'),
                  calibrat3_frames_with_board_per_camera=c3t.get('detectionPerView'),
                  anipose_frames_with_board_per_camera=aps.get('frames_per_cam'))
    return spec, arms, dict(frames=frames,date=spec.get('date'), note=spec.get('note', ''), tuning=bool(spec.get('tuning')),
                            arms={k: arm_record(v) for k, v in arms.items()},
                            scoring_set=sc['anipose']['_meta'] if sc else None,
                            secondary_set_calibrat3_corners={k: v['overall'] for k, v in (sc or {}).get('calibrat3', {}).items() if k != '_meta'},
                            anipose_arm=jl(os.path.join(d, 'anipose_arm.json')),
                            anipose_repeats=jl(os.path.join(d, 'anipose_repeats.json')),
                            controls=summarize_controls(d),
                            input_sha256=spec.get('sha256'))


def rig(prefix, size, timing_key):
    recs, pooled, cams = {}, {}, None
    for d in sorted(glob.glob(os.path.join(W, f'{prefix}*'))):
        if not os.path.exists(os.path.join(d, 'scores_anipose.npz')):
            continue
        sid = os.path.basename(d)[len(prefix):] or os.path.basename(d)
        spec, arms, r = recording(d)
        cams = [c[0] for c in spec['cameras']]
        recs[sid] = r
        for k, v in arms.items():
            for cam, e in v.items():
                pooled.setdefault(k, {}).setdefault(cam, []).append(e)
    if not recs:
        return None
    e = dict(cameras=cams, size=size, n_cameras=len(cams), recordings=recs,
             detsets={'anipose': dict(arms={k: arm_record({c: np.concatenate(v) for c, v in dd.items()})
                                            for k, dd in pooled.items()},
                                      meta=dict(recordings=len(recs), held_out=sum(not r['tuning'] for r in recs.values())))})
    t = timing_block(os.path.join(W, timing_key))
    if t:
        e['timing'] = t
        e['timing_recording'] = timing_key
    return e


rec = dict(hist_edges=EDGES.tolist(), datasets={}, opencv='4.13.0 (both detectors)', variant='b')
for key, prefix, size, tkey in (('slap8', 'slap_', [1280, 1024], 'slap_10072022120554'),
                                ('md10', 'md10_', [1280, 1024], 'md10_20250514_134355')):
    e = rig(prefix, size, tkey)
    if e:
        rec['datasets'][key] = e
        print(f'{key}: {len(e["recordings"])} recordings, timing={"timing" in e}')

d = os.path.join(W, 'calib18')
spec, arms, r = recording(d)
c18 = dict(cameras=[c[0] for c in spec['cameras']], size=[1680, 1200], n_cameras=len(spec['cameras']),
           recording_detail=r,   # not `recordings`: the published calib18 entry has none, panels draw per-recording lines from it
           detsets={'anipose': dict(arms=r['arms'], meta=dict(recordings=1, held_out=0))})
abl = load_npz(os.path.join(d, 'ablation', 'scores_anipose.npz')) if os.path.exists(os.path.join(d, 'ablation', 'scores_anipose.npz')) else {}
ho = jl(os.path.join(d, 'heldout.json'), {})
if abl:
    c18['ablation'] = dict(
        all={(k if k == 'anipose' else f'abl_{k}'): float(np.median(np.concatenate(list(v.values()))))
             for k, v in abl.items() if k != 'calibrat3_800'},
        heldout={(k if k == 'anipose' else f'abl_{k}'): v['heldout']['median']
                 for k, v in (ho.get('scores') or {}).items() if isinstance(v, dict) and 'heldout' in v},
        heldout_meta={k: ho[k] for k in ('frames_used_by_calibrat3', 'frames_total', 'frames_heldout') if k in ho},
        scoring_set=(jl(os.path.join(d, 'ablation', 'scores.json')) or {}).get('anipose', {}).get('_meta'),
        note='ablation arms scored on their OWN shared set (ablation/), not the E/F set')
t = timing_block(d)
if t:
    c18['timing'] = t
    c18['timing_recording'] = 'calib18'
rec['datasets']['calib18'] = c18


def gt_rows(txt):
    rows = {}
    for line in txt.splitlines():
        p = line.split()
        if len(p) == 9 and p[1].endswith('p'):
            rows[p[0]] = dict(d_pp=float(p[1][:-1]), d_fx=float(p[2][:-1]), d_fy=float(p[3][:-1]),
                              d_k1=float(p[4]), d_k2=float(p[5]), centre_mm=float(p[6][:-2]),
                              scale_pct=float(p[7][:-1]), centre_pct=float(p[8][:-1]))
    return rows


syn = {}
for gt in ('full', 'anipose'):
    sd = os.path.join(W, f'synth_{gt}')
    if not os.path.exists(os.path.join(sd, 'calibrat3_800.toml')):
        continue
    txt = subprocess.run([sys.executable, f'{FIGS}/fig7_calib_gt_compare.py', os.path.join(bench, 'inputs', f'synth_{gt}', 'calibration_gt.toml'), '--',
                          f'calibrat3={sd}/calibrat3_800.toml', f'anipose={sd}/calibration_anipose.toml'],
                         capture_output=True, text=True).stdout
    syn[gt] = dict(arms=gt_rows(txt), raw=txt)
if 'full' in syn:
    rec['synthetic'] = dict(arms=syn['full']['arms'], raw=syn['full']['raw'],
                            within_anipose_model=syn.get('anipose'))

git = lambda repo: subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
rec['provenance'] = dict(
    bench=os.path.abspath(bench),
    calibrat3_sha=git('/root/vast/eric/sleap-3d-gui/scratch/repos/calibrat3'),
    lucid_sha=git(FIGS),
    env=jl(os.path.join(bench, 'logs', 'env.json')),
    transcode=open(os.path.join(bench, 'logs', 'x264_args.txt')).read() if os.path.exists(os.path.join(bench, 'logs', 'x264_args.txt')) else None)
json.dump(rec, open(out_path, 'w'))
print(f'wrote {out_path} ({os.path.getsize(out_path) / 1e3:.0f} kB)')
