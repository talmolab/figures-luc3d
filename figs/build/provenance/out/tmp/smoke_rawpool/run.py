# Smoke test: track + score the no-rule pool exactly as mdb_experiment.py b3 (tail in tracking, tail-free scoring).
import json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
FIGS = Path("/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs"); sys.path.insert(0, str(FIGS))
BM = Path("/root/vast/eric/luc3d-bench/outputs/bmimica"); POOL = BM / "det_h5_raw_smoke"
CAMS = ["21241563", "21369048", "21372315", "21372316", "22085397"]
T = FIGS / "out/tmp/smoke_rawpool"
S = ["20250907_174343", "20250907_143021", "20250904_140306", "20250908_161812", "20250829_162253"]
def one(s):
    o = T / f"{s}.json"
    cal = next((Path("/root/vast/eric/BMimica") / s / "calibration").glob("*_calibration.toml"))
    r = subprocess.run(["node", str(FIGS / "fig3-bench/fig3_bench.mjs"), "--session-idx", "0", "--num-animals", "2",
                        "--calibration", str(cal), "--pred-h5-dir", str(POOL / s), "--cameras", ",".join(CAMS),
                        "--out", str(o), "--no-exclude-tail"], capture_output=True, text=True)
    if r.returncode: return s, "FAILED " + r.stderr[-300:]
    import fig3_score as fs
    sc = fs.score_session(str(o), str(POOL / s), str(BM / "gt" / s), CAMS, 2)
    return s, {k: sc[k] for k in ("cross_idf1", "cross_idp", "cross_idr", "within_switches")}
if __name__ == "__main__":
    assert os.environ.get("SCORE_NOTAIL") == "1" and os.environ.get("LUCID_ROOT")
    ref = json.load(open(FIGS / "out/mdb_experiment_v105.json"))["configs"]
    res = dict(ProcessPoolExecutor(5).map(one, S))
    json.dump(res, open(T / "results.json", "w"), indent=1)
    print(f"{'session':16s} {'b1 (body>=5, no tail)':>22s} {'b3 (body>=5, tail)':>22s} {'NO RULE (tail)':>22s}")
    for s in S:
        r = res[s]
        if isinstance(r, str): print(s, r); continue
        f = lambda c: f"{ref[c]['per_session'][s]['cross_idf1']:.4f} / {ref[c]['per_session'][s]['within_switches']:>3}sw"
        print(f"{s:16s} {f('b1'):>22s} {f('b3'):>22s} {r['cross_idf1']:.4f} / {r['within_switches']:>3}sw")
    ok = [s for s in S if not isinstance(res[s], str)]
    m = lambda c: sum(ref[c]['per_session'][s]['cross_idf1'] for s in ok) / len(ok)
    print(f"{'MEAN':16s} {m('b1'):>22.4f} {m('b3'):>22.4f} {sum(res[s]['cross_idf1'] for s in ok)/len(ok):.4f}")
