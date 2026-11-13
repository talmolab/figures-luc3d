import sys
from pathlib import Path
sys.path.insert(0, "scripts")
import run_bytetrack_bench as bt
SERIALS = ["21241563","21369048","21372315","21372316","22085397"]
bt.BENCH_CAMERAS = tuple(SERIALS)
out_dir = "outputs/bartul_smoke/results/bytetrack"
bt.process_session((0, "20250827_141755", True, "outputs/bartul_smoke/det_h5", out_dir))
print("bytetrack done ->", out_dir)
