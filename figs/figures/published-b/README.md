# The published figures (b series), kept for comparison

The seven composites as placed in the manuscript, copied from `eric/build-wrappers`
`b0a4c34` (byte-identical to the Overleaf figures, 2026-11-13). On this branch
`figures/fig1`-`fig7` hold the q series instead, the same figures with Mouse-Dyad session
20250904_183921 left out. Figs 1 and 4 and Supp Fig 2 are pixel-identical between the two;
Figs 2, 3 and 5 and Supp Fig 4 differ.

The b deposits (`out/*_b.*`) are unchanged on this branch, so the b figures can still be
rebuilt here: `FIG_SERIES=b bash figs/build/run_FINALBUILD.sh` (the default is q).
