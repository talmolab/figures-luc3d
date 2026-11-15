import sys
sys.argv = ["swap_dp",
    "--in-dir", "outputs/bartul_smoke/results/liezl_c6_trunc",
    "--h5-dir", "outputs/bartul_smoke/det_h5",
    "--master", "outputs/bartul_smoke/bartul_master.tsv",
    "--out-dir", "outputs/bartul_smoke/results/swapdp_c6"]
SERIALS = ["21241563","21369048","21372315","21372316","22085397"]
import scripts.temporal_link_3d as tl
tl.CAMS = SERIALS
import scripts.swap_dp as sd
sd.CAMS = SERIALS
sd.main()
