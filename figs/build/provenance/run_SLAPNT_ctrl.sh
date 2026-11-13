#!/usr/bin/env bash
# Gate for B2: the uv sleap_nn_env (sleap-nn 0.2.0, py 3.11.14) must reproduce the deposited
# SLAP-2M SLEAP arm (sleap_cap/, made with the same packages under miniforge py 3.11.10) exactly.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
FAIR_VARIANT=bvis5 FAIR_SLEAP_PY=/root/vast/eric/luc3d-bench/sleap_nn_env/bin/python FAIR_SLEAP_OUT_DIR=figs/out/tmp/slap2m_fair_bvis5/ctrl_sleap_cap_uv /root/vast/eric/luc3d-bench/sleap_nn_env/bin/python figs/fig6_slap2m_fair.py --stage sleap --sessions 10192022180019 --workers 6 > figs/out/md56_logs/SLAPNT_ctrl.log 2>&1
echo "track exit=$?"
/root/vast/eric/luc3d-bench/sleap_nn_env/bin/python - <<'PY'
import sleap_io as sio, numpy as np, os, sys
A='figs/out/tmp/slap2m_fair_bvis5/sleap_cap/10192022180019'; B='figs/out/tmp/slap2m_fair_bvis5/ctrl_sleap_cap_uv/10192022180019'
ok=bad=0
for f in sorted(x for x in os.listdir(B) if x.endswith('.slp') and 'partial' not in x):
    def rows(p):
        L=sio.load_slp(p); return {lf.frame_idx: sorted((i.track.name if i.track else '', np.nan_to_num(i.numpy(),nan=-1e9).round(6).tobytes()) for i in lf.instances) for lf in L.labeled_frames}
    ra,rb=rows(f'{A}/{f}'),rows(f'{B}/{f}'); d=[k for k in set(ra)|set(rb) if ra.get(k)!=rb.get(k)]
    print(f, len(ra), len(rb), 'differ', len(d)); ok+= not d; bad+= bool(d)
print('CONTROL', 'PASS' if bad==0 and ok==6 else 'FAIL', ok, bad); sys.exit(0 if bad==0 and ok==6 else 1)
PY
echo "SLAPNT CTRL exit=$?"
