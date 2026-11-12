# Supplementary videos

The four supplementary videos, as shown on https://talmolab.github.io/luc3d-page/.
All H.264, 1920 × 1080, silent, MP4 with `+faststart`.

| File | Video | Length | Source |
|---|---|---|---|
| `supp_video1_slap2m_flythrough.mp4` | **1. SLAP-2M.** Home-cage flythrough: the enclosure and its cameras in 3D, four mice tracked with per-animal identity colors, and the skeletons projected onto one camera's footage | 13 s | Blender render of a SLAP-2M session |
| `supp_video2_mousedyad10m_social.mp4` | **2. Mouse-Dyad-10M.** The 3D poses of a male (blue) and a female (red) from one session | 14 s | `drafts/videos/supp_video2.mp4` (2026-10-29, copied as is) |
| `supp_video3_slap2m_difficulty.mp4` | **3. SLAP-2M difficulty.** Every recorded combination of animal count (1–4) and enrichment rating, one session each, in 3D, played at 2× | 63 s | `blender-images/slap2m_movie.py` v4 (2026-11-12): `compose` from the v2 renders plus 4 re-rendered segments |
| `supp_video4_walkthrough.mp4` | **4. Walkthrough.** The full LUC3D v1.0.5 workflow in the browser, from loading videos, the arena floor and a skeleton, through labeling, Track All and its summary with the automatic ID-switch check, Triangulate All, assigning IDs in the Instances panel and the 3D view | 10 min 9 s | `drafts/videos/walkthrough_full_draft11_v6.mp4` (2026-10-29, copied as is) |

Videos 2 and 4 were remuxed from their sources without re-encoding (`-c copy -movflags +faststart`).

**Video 3, v4 (2026-11-12).** v3 stepped 2 data frames per output frame (`DATA_STEP = 2`) in
every segment, so the four segments from the 60 fps `2022-10-19` sessions, conditions
(1, 0), (1, 1), (1, 3) and (2, 3), played at 1× (real time) while the caption and the
on-screen label said 2×. v4 steps 4 data frames per output frame for those sessions
(`step_for()`), re-picks their windows over the doubled span (same sessions; (1, 3) and
(2, 3) move to new start frames that pass the same floor and clearance checks), and
re-renders them (bpy 5.2.0, Cycles, 24 samples, RTX A6000). The other ten segments,
the cards and the remaining cut-ins reuse the v2 renders; outside the four segments
the v4 file matches v3 at ≥ 45 dB PSNR. Same length (1,884 frames, 62.8 s), QC clean.
The per-combo windows are in `blender-images/renders/movie3/picks/`.
