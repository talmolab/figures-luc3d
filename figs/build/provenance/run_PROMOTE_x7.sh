#!/usr/bin/env bash
# Promote the x7 set (2026-10-20, eric-mimicc-0-1) to the main figures/fig{2..7} + data/fig{2..7},
# as commit 01fb3530 promoted the b series: rebuild from the SAME inputs (FIG_VARIANT=x7, every link
# required, same env), no FIG_TAG, then assert every data CSV is byte-identical to its x7 twin.
# The previous main figures are backed up first.
set -uo pipefail
R=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid; cd $R/figs
L=out/md56_logs; FPY=.venv/bin/python
fail() { echo "PROMOTE FAILED: $*"; exit 1; }
grep -q "X7 BUILD DONE" $L/run_X7_build.out || fail "x7 build not done"
B=$L/main_before_x7_2026-10-20; [ -e $B ] && fail "$B exists"
mkdir -p $B && for n in 2 3 4 5 6 7; do cp -rp figures/fig$n $B/figures_fig$n; cp -rp data/fig$n $B/data_fig$n; done

export FIG6F_NOEVICT=stale0_dist25 SIGN_TIES_EXCLUDED=1 FIG5G_PERDIR=1 F3_TRUE_MEDIAN=1 FIG3F_CLEAN=1 \
       FIG6_SDANNCE_AGG=$R/figs/fig10-bench/results_v105/agg
export FIG_VARIANT=x7 FIG_VARIANT_LOG=$PWD/$L/PROMOTE_loads.tsv
export FIG_VARIANT_REQUIRE=$(cut -f1 $L/X7_links.tsv | paste -sd, -)
unset FIG_TAG
rm -f $L/PROMOTE_loads.tsv
mkdir -p figures/fig6
$FPY panels/fig6slapstale_reduction.py > $L/PROMOTE_fig6f.log 2>&1 || fail "fig6 f"
for n in 2 3 4 5 6 7; do
  $FPY make_figures.py $n > $L/PROMOTE_fig$n.log 2>&1; E=$?
  NF=$(grep -c "    ! " $L/PROMOTE_fig$n.log || true)
  echo "fig$n exit=$E failed_panels=$NF $(tail -n1 $L/PROMOTE_fig$n.log)"
  [ $E -eq 0 ] && [ "$NF" = 0 ] || fail "fig$n"
done
# the unplaced non-variant within-vs-cross panel, as every b / c build drops it
rm -f figures/fig6/fig6a_within_vs_cross.pdf figures/fig6/fig6a_within_vs_cross.png data/fig6/fig6a_within_vs_cross.csv
echo "--- data CSV vs x7 twin (must be identical):"
bad=0
for n in 2 3 4 5 6 7; do
  for f in data/fig${n}x7/*.csv; do b=$(basename $f); [ "$b" = fig6a_within_vs_cross.csv ] && continue
    cmp -s $f data/fig$n/$b || { echo "DIFF fig$n/$b"; bad=1; }; done
done
[ $bad -eq 0 ] || fail "main CSVs differ from x7"
echo "PROMOTE DONE (backup $B)"
