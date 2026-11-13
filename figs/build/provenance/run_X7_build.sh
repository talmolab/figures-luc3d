#!/usr/bin/env bash
# X7 (2026-10-20, eric-mimicc-0-1): ONE variant (`x7`) and ONE tag (`x7`) for Figs 2-7.
# Builds into figures/fig{2..7}x7 + data/fig{2..7}x7 only; nothing current is overwritten.
# Pass 1 logs what main Figs 4, 5, 7 read under their original variants (md56 / final /
# none) into a throwaway tag. The x7 link set is then the union of every deposit the
# current figures read -- fig2c (FIG2C_loads), fig3c (FIG3C_clean3F_loads), fig6c
# (M3_loads), main fig4/5/7 (pass 1) -- each pointing at the file it resolved to then.
# A name that resolved to two different files is a conflict and stops the build.
# Pass 2 builds everything under FIG_VARIANT=x7 with FIG_VARIANT_REQUIRE = the whole
# link set, then the gates and a CSV diff against the current figures.
set -uo pipefail
R=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid; cd $R/figs
L=out/md56_logs; O=out; FPY=.venv/bin/python
fail() { echo "X7 FAILED: $*"; exit 1; }
for t in x7 x7pre; do for n in 2 3 4 5 6 7; do
  { [ -e figures/fig${n}${t} ] || [ -e data/fig${n}${t} ]; } && fail "figures/fig${n}${t} or data/fig${n}${t} exists"
done; done
ls $O/*_x7.json >/dev/null 2>&1 && fail "x7 links exist already"

export FIG6F_NOEVICT=stale0_dist25 SIGN_TIES_EXCLUDED=1 FIG5G_PERDIR=1 F3_TRUE_MEDIAN=1 FIG3F_CLEAN=1 \
       FIG6_SDANNCE_AGG=$R/figs/fig10-bench/results_v105/agg

# --- pass 1: what main Figs 4, 5, 7 read
rm -f $L/X7pre_loads.tsv
for spec in 4:md56 5:final 7:; do n=${spec%%:*}; v=${spec#*:}
  FIG_TAG=x7pre FIG_VARIANT=$v FIG_VARIANT_LOG=$PWD/$L/X7pre_loads.tsv \
    $FPY make_figures.py $n --panels > $L/X7pre_fig$n.log 2>&1 || fail "pass 1 fig$n"
done
rm -rf figures/fig{4,5,7}x7pre data/fig{4,5,7}x7pre

# --- the x7 link set
$FPY - <<'PY' || fail "link set"
import os, sys
O = "out"; L = "out/md56_logs"
m = {}
# LUC3D-JS plan (2026-10-20): Fig 2 from the LUC3D-built fig2c (FIG2C_js_loads.tsv), Fig 3
# from whatever fig3c build eric-mimic2 last logged (X7_FIG3_LOADS), and explicit overrides
# for deposits rebuilt on LUC3D since those logs (X7_OVERRIDE="a.json=b.json,...").
for f in ("FIG2C_js_loads.tsv", os.environ.get("X7_FIG3_LOADS", "FIG3C_clean3F_loads.tsv"),
          "M3_loads.tsv", "X7pre_loads.tsv"):
    for line in open(f"{L}/{f}"):
        _s, name, got = line.rstrip("\n").split("\t")
        real = os.path.relpath(os.path.realpath(f"{O}/{got}"), os.path.realpath(O))
        if m.setdefault(name, (real, f))[0] != real:
            sys.exit(f"CONFLICT {name}: {m[name][0]} ({m[name][1]}) vs {real} ({f})")
for kv in filter(None, os.environ.get("X7_OVERRIDE", "").split(",")):
    name, tgt = kv.split("=")
    if not os.path.exists(f"{O}/{tgt}"):
        sys.exit(f"override target {tgt} missing")
    real = os.path.relpath(os.path.realpath(f"{O}/{tgt}"), os.path.realpath(O))
    print(f"override {name}: {m.get(name, ('-',))[0]} -> {real}")
    m[name] = (real, "X7_OVERRIDE")
with open(f"{L}/X7_links.tsv", "w") as fh:
    for name, (real, f) in sorted(m.items()):
        stem, ext = os.path.splitext(name)
        os.symlink(real, f"{O}/{stem}_x7{ext}")
        fh.write(f"{name}\t{real}\t{f}\n")
print(f"{len(m)} x7 links")
PY
export FIG_VARIANT=x7 FIG_TAG=x7 FIG_VARIANT_LOG=$PWD/$L/X7_loads.tsv
export FIG_VARIANT_REQUIRE=$(cut -f1 $L/X7_links.tsv | paste -sd, -)
rm -f $L/X7_loads.tsv

# --- pass 2. Fig 6 f is panels/fig6slapstale_reduction.py (not in make_figures' glob);
# it writes the axes json fig6_09 copies, so it runs first.
mkdir -p figures/fig6x7
$FPY panels/fig6slapstale_reduction.py > $L/X7_fig6f.log 2>&1 || fail "fig6 f"
for n in 2 3 4 5 6 7; do
  $FPY make_figures.py $n > $L/X7_fig$n.log 2>&1; E=$?
  NF=$(grep -c "    ! " $L/X7_fig$n.log || true)
  echo "fig${n}x7 exit=$E failed_panels=$NF $(tail -n1 $L/X7_fig$n.log)"
  [ $E -eq 0 ] && [ "$NF" = 0 ] || fail "fig$n (see $L/X7_fig$n.log)"
done
echo "--- loads that did NOT come from an x7 link:"
awk -F'\t' '$3 !~ /_x7\./' $L/X7_loads.tsv | cut -f2,3 | sort | uniq -c

# --- gates (S4 rerun, M6)
SCORE_NOTAIL=1 $FPY s4_frame_gate.py > $L/X7_s4.log 2>&1; echo "S4 exit=$? $(tail -1 $L/X7_s4.log)"
$FPY m6_frame_gate.py > $L/X7_m6.log 2>&1; echo "M6 exit=$? $(tail -1 $L/X7_m6.log)"

# --- CSV diff against what the paper shows now
for spec in 2:2c 3:3c 4:4 5:5 6:6c 7:7; do n=${spec%%:*}; cur=data/fig${spec#*:}; new=data/fig${n}x7
  echo "=== fig$n: $new vs $cur"
  diff -rq $cur $new
done
echo "X7 BUILD DONE"
