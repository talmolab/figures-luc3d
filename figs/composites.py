#!/usr/bin/env python3
"""
Composites that place their panels by hand rather than through assemble.LAYOUTS:
Figs 1, 2, 3 and 6 (Supp Fig 4). assemble.main() hands these four figure numbers here.
They were the 2026-10-27 layout candidates fig1b / fig3b / fig2s / fig6b (Eric) and
became the main figures on 2026-10-29 with the Fig 2 <-> Fig 3 swap, so the repo's
fig2 is now the tracker figure and fig3 the labeling figure, as in the paper.

  fig1  a = per-camera tracks / LUC3D identities / 3D + cam 0 reprojection as one
        block (rows = stages, columns = cam 0 / cam 7), with the ID 1 flow marks;
        b = the two arena renders stacked beside it; c = the pipeline.
  fig2  a = the tracker's 2D/3D anchor diagram (2D over 3D), b = hypothesis
        illustration, c = per-frame loop strip (panels/fig2_19_algorithm_strip.py)
        beside d = exhaustive hypotheses per frame
        (panels/fig2_20_cost_model_third.py), e = identity switch, f-i = quality,
        time per frame and the two r sweeps. The panel FILES keep their pre-swap
        letters (fig2b_hyp_illustration is b, fig2e_idswitch is e, ...).
  fig3  the labeling figure, row for row as assemble.LAYOUTS[3] places it.
  fig6  Supp Fig 4: the a-d tracker block beside the s-DANNCE survival (e, file
        fig6g), then the staleness sweep (f).

Nothing is re-measured. Panels are embedded as vector (PyMuPDF show_pdf_page,
clipped where a tile is cut out of a row). The anchor halves are
`figures/drafts/figs/lucid_chen2020_style_{2d,3d}.png`, written by
`fig_chen2020_style.py --split`.

    .venv/bin/python fig_chen2020_style.py --split
    .venv/bin/python composites.py          # 1, 2, 3, 6
    .venv/bin/python composites.py 2        # just Fig 2
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz

FIGS = Path(__file__).resolve().parent
sys.path.insert(0, str(FIGS))
from assemble import (GUTTER, INK, LETTER_LEAD, LETTER_PT, MARGIN, MM, PAGE_W,  # noqa: E402
                      PREVIEW_DPI, ROW_GAP, TITLE_PT, TITLES, axes_extent)

FIGURES = FIGS / "figures"
ANCHOR_2D = FIGURES / "drafts" / "figs" / "lucid_chen2020_style_2d.png"
ANCHOR_3D = FIGURES / "drafts" / "figs" / "lucid_chen2020_style_3d.png"
ANCHOR_GAP = 1.5          # mm between the 2D and 3D halves
LEAD = ROW_GAP + LETTER_LEAD


def png_size(p: Path) -> tuple[int, int]:
    pm = fitz.Pixmap(str(p))
    return pm.width, pm.height


def pdf_size(p: Path) -> tuple[float, float]:
    d = fitz.open(p)
    r = d[0].rect
    d.close()
    return r.width / MM, r.height / MM


class Sheet:
    """One composite page. All coordinates in mm from the top-left."""

    def __init__(self, h_mm: float):
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=PAGE_W * MM, height=h_mm * MM)
        self.h = h_mm
        # a uniquely named font -- see fig2_sync.build_stack on why not "hebo"
        self.page.insert_font(fontname="VarHeboBold",
                              fontbuffer=fitz.Font("Helvetica-Bold").buffer)
        self.page.insert_font(fontname="VarHelv",
                              fontbuffer=fitz.Font("Helvetica").buffer)

    @staticmethod
    def R(x, y, w, h):
        return fitz.Rect(x * MM, y * MM, (x + w) * MM, (y + h) * MM)

    def pdf(self, src: Path, x, y, w, h=None, clip=None):
        """Place `src` (or its `clip` = (x0, y0, x1, y1) mm) at width w, uniform."""
        d = fitz.open(src)
        cr = self.R(clip[0], clip[1], clip[2] - clip[0], clip[3] - clip[1]) if clip else None
        sw, sh = ((clip[2] - clip[0], clip[3] - clip[1]) if clip
                  else (d[0].rect.width / MM, d[0].rect.height / MM))
        h = h if h is not None else w * sh / sw
        self.page.show_pdf_page(self.R(x, y, w, h), d, 0, clip=cr)
        d.close()
        return h

    def png(self, src: Path, x, y, w):
        pw, ph = png_size(src)
        h = w * ph / pw
        self.page.insert_image(self.R(x, y, w, h), filename=str(src))
        return h

    def letter(self, letter, x, y, title=None, axes_pdf=None, w=None):
        """Letter hanging 2 mm left of a panel at (x, y), title beside it --
        centred over the panel's axes when `axes_pdf` has ticked spines."""
        lx = max(0.6, x - 2.0)
        self.page.insert_text(fitz.Point(lx * MM, (y - 1.2) * MM), letter,
                              fontname="VarHeboBold", fontsize=LETTER_PT, color=INK)
        if not title:
            return
        tx = lx + 4.4
        if axes_pdf is not None and w is not None:
            span = axes_extent(axes_pdf)
            if span:
                k = w / pdf_size(axes_pdf)[0]
                tw = fitz.Font("Helvetica-Bold").text_length(title, fontsize=TITLE_PT) / MM
                tx = min(max(x + (span[0] + span[1]) / 2 * k - tw / 2, tx), x + w - tw)
        self.page.insert_text(fitz.Point(tx * MM, (y - 1.2) * MM), title,
                              fontname="VarHeboBold", fontsize=TITLE_PT, color=INK)

    def text(self, s, x, y, size=7.5, color=(0.35, 0.35, 0.35), center=True, bold=False):
        f = fitz.Font("Helvetica-Bold" if bold else "Helvetica")
        tw = f.text_length(s, fontsize=size) / MM
        self.page.insert_text(fitz.Point(((x - tw / 2) if center else x) * MM, y * MM), s,
                              fontname="VarHeboBold" if bold else "VarHelv",
                              fontsize=size, color=color)

    def save(self, name: str):
        out = FIGURES / name / f"{name}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(out, deflate=True)
        self.doc[0].get_pixmap(dpi=PREVIEW_DPI).save(out.with_suffix(".png"))
        self.doc.close()
        print(f"  wrote {out.relative_to(FIGS)} (+ .png)  180 x {self.h:.0f} mm")


def _trimmed(p: Path, pad=6, band=18) -> "fitz.Pixmap":
    """`p` with its white margins cut to `pad` px and any blank horizontal band
    inside it taller than 2 * band (the gap under the "2D/3D correspondence"
    heading) cut to `band` px -- the type keeps its size, the white goes."""
    import numpy as np
    pm = fitz.Pixmap(str(p))
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)
    ink = (a[..., :3] < 240).any(2)
    xs, ys = np.where(ink.any(0))[0], np.where(ink.any(1))[0]
    x0, x1 = max(0, xs[0] - pad), min(pm.width, xs[-1] + 1 + pad)
    y0, y1 = max(0, ys[0] - pad), min(pm.height, ys[-1] + 1 + pad)
    rows, keep, run = ink[:, x0:x1].any(1), [], []
    for i in range(y0, y1):
        if rows[i]:
            keep += run if len(run) <= 2 * band else run[:band // 2] + run[-band // 2:]
            keep.append(i)
            run = []
        else:
            run.append(i)
    keep += run
    b = np.ascontiguousarray(a[keep][:, x0:x1])
    cs = fitz.csRGB if pm.n - pm.alpha == 3 else fitz.csGRAY
    return fitz.Pixmap(cs, b.shape[1], b.shape[0], b.tobytes(), pm.alpha)


def _anchor_pixmaps(trim):
    if trim:
        return _trimmed(ANCHOR_2D), _trimmed(ANCHOR_3D)
    return fitz.Pixmap(str(ANCHOR_2D)), fitz.Pixmap(str(ANCHOR_3D))


def anchor_stack(sheet: Sheet, x, y, col_w, trim=False):
    """2D over 3D, both at one scale so their type matches; returns height.
    `trim` cuts the PNGs' white margins and heading gap first (Fig 2)."""
    p2, p3 = _anchor_pixmaps(trim)
    (w2, h2), (w3, h3) = (p2.width, p2.height), (p3.width, p3.height)
    s = col_w / max(w2, w3)                      # mm per px
    sheet.page.insert_image(Sheet.R(x + (col_w - w2 * s) / 2, y, w2 * s, h2 * s), pixmap=p2)
    sheet.page.insert_image(Sheet.R(x + (col_w - w3 * s) / 2, y + h2 * s + ANCHOR_GAP,
                                    w3 * s, h3 * s), pixmap=p3)
    return (h2 + h3) * s + ANCHOR_GAP


def anchor_h_per_w(trim=False) -> float:
    p2, p3 = _anchor_pixmaps(trim)
    return (p2.height + p3.height) / max(p2.width, p3.width)


def content_box(p: Path, pad=0.8, sides="tb"):
    """`p`'s page box (mm) shrunk to its ink plus `pad` on the named sides only
    ("tb" keeps the full width, so axes_extent-centred titles still line up)."""
    import numpy as np
    pg = fitz.open(p)[0]
    pm = pg.get_pixmap(dpi=200)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)[..., :3]
    ink = (a < 245).any(2)
    xs, ys = np.where(ink.any(0))[0], np.where(ink.any(1))[0]
    k = 25.4 / 200
    W, H = pg.rect.width / MM, pg.rect.height / MM
    return (max(0, xs[0] * k - pad) if "l" in sides else 0,
            max(0, ys[0] * k - pad) if "t" in sides else 0,
            min(W, (xs[-1] + 1) * k + pad) if "r" in sides else W,
            min(H, (ys[-1] + 1) * k + pad) if "b" in sides else H)


# ---------------------------------------------------------------------------
def fig2():
    d = FIGURES / "fig2"
    # c = the per-frame loop strip (panels/fig2_19_algorithm_strip.py), full width,
    # placed AFTER the anchor diagram (a) that defines what an anchor is and the
    # hypotheses (b) it replaces (Eric, 2026-10-27)
    # d = the exhaustive hypothesis explosion (base Fig 3's old b) beside it, at a
    # third span (panels/fig2_20_cost_model_third.py); the strip is its compact form
    strip = d / "fig2c_algorithm_strip.pdf"
    cost = d / "fig2d_cost_model_third.pdf"
    for p_, script in ((strip, "fig2_19_algorithm_strip.py"),
                       (cost, "fig2_20_cost_model_third.py")):
        if not p_.exists():
            sys.exit(f"missing {p_} -- run panels/{script} first")
    # LESS WHITESPACE (Eric, 2026-10-28, so Fig 3 + legend fit one page): every
    # panel is placed by its INK box (content_box), not its page box, and a's two
    # PNGs lose their margins and heading gap (_trimmed). That lets row 1 drop
    # from 92 to ROW1_H mm with a's type no smaller than before; b prints ~5%
    # smaller. Cells keep their full width so axes-centred titles still line up.
    cs_, cc_ = content_box(strip), content_box(cost)
    stw, cow = pdf_size(strip)[0], pdf_size(cost)[0]
    sh = max(cs_[3] - cs_[1], cc_[3] - cc_[1])
    hyp = d / "fig2b_hyp_illustration.pdf"
    hb_ = content_box(hyp, sides="tblr")
    k = anchor_h_per_w(trim=True)
    ROW1_H, ROW3_SCALE = 82.0, 0.90
    H1 = ROW1_H
    W = (H1 - ANCHOR_GAP) / k
    hyp_w = (hb_[2] - hb_[0]) * H1 / (hb_[3] - hb_[1])
    idsw = d / "fig2e_idswitch.pdf"
    ib_ = content_box(idsw, sides="tblr")
    iw, ih = (ib_[2] - ib_[0]) * ROW3_SCALE, (ib_[3] - ib_[1]) * ROW3_SCALE
    cells = [("f", "quality"), ("g", "head_to_head"),
             ("h", "sweep_switches"), ("i", "sweep_idf1")]
    srcs = [d / f"fig2{letter}_{slug}.pdf" for letter, slug in cells]
    cbs = [content_box(p_) for p_ in srcs]
    cw = pdf_size(srcs[0])[0] * ROW3_SCALE
    chs = [(cb[3] - cb[1]) * ROW3_SCALE for cb in cbs]
    row3_h = max(ih, max(chs[0], chs[2]) + LEAD + max(chs[1], chs[3]))
    page_h = MARGIN + LETTER_LEAD + sh + LEAD + H1 + LEAD + row3_h + MARGIN
    s = Sheet(page_h)

    y = MARGIN + LETTER_LEAD
    x1 = (PAGE_W - (W + GUTTER + hyp_w)) / 2
    anchor_stack(s, x1, y, W, trim=True)
    s.letter("a", x1, y, TITLES[(2, "a")])
    s.pdf(hyp, x1 + W + GUTTER, y, hyp_w, H1, clip=hb_)
    s.letter("b", x1 + W + GUTTER, y, TITLES[(2, "b")])
    y += H1 + LEAD
    s.pdf(strip, 0, y, stw, clip=cs_)
    s.letter("c", 0, y, TITLES[(2, "c")])
    s.pdf(cost, stw + GUTTER, y, cow, clip=cc_)
    s.letter("d", stw + GUTTER, y, TITLES[(2, "d")], cost, cow)

    y += sh + LEAD
    x3 = (PAGE_W - (iw + 2 * GUTTER + 2 * cw)) / 2
    s.pdf(idsw, x3, y, iw, clip=ib_)
    s.letter("e", x3, y, TITLES[(2, "e")])
    top_h = max(chs[0], chs[2])
    for i, (letter, slug) in enumerate(cells):
        cx = x3 + iw + GUTTER + (i // 2) * (cw + GUTTER)
        cy = y + (i % 2) * (top_h + LEAD)
        s.pdf(srcs[i], cx, cy, cw, chs[i], clip=cbs[i])
        s.letter(letter, cx, cy, TITLES.get((2, letter)), srcs[i], cw)
    s.save("fig2")
    print(f"    anchor column {W:.1f} x {H1:.1f} mm, hyp {hyp_w:.1f} mm wide, "
          f"row 3 x{ROW3_SCALE}")


# ---------------------------------------------------------------------------
#: Fig 1's flow marks (Eric, 2026-10-27: "like we do in fig3c, draw lines ... from
#: the track color to the same id color, then those same id colors to that 3D pose,
#: then from that 3D pose to the reprojections"). Same vocabulary as Fig 3c/3d
#: (idswitch_fig_style): DASHED colour-GRADIENT legs where a mark changes identity
#: colour (track -> ID), SOLID identity-coloured legs where it does not (ID -> 3D).
#: Every leg sits on a white halo so it reads over the grayscale photos. Labels
#: (fig1a_tracking.pdf's own text spans) only pick WHICH instance; legs and colours
#: come from the instances themselves.
FLOW_LW = 1.6 * 25.4 / 72          # mm, = idswitch ASSOC_LW (1.6 pt)
DASH, GAPL = 1.3, 0.9              # mm
HEAD = 3.6                         # mm, arrowhead length (1.7 -> 2.8 -> 3.6: "bigger")
CHIP_PAD = 0.55                    # mm (source units) a label chip extends past its text
CLEAR = 0.45                       # mm kept between any leg/head and a label chip
HEAD_CLEAR = 0.2                   # mm, the same for a track -> ID head passing a chip
DOT_STEP = 1.05                    # mm between dots on the reprojection leg
#: (track label, ID label) per camera column, same animal. ID 1 ONLY (Eric,
#: 2026-10-27: "just show the ID1 tracks ... the other arrows [are] confusing") --
#: one animal followed from its two track names to one identity, one 3D pose and
#: its reprojection.
TRACK_TO_ID = {0: [("T 89", "ID 1")], 1: [("T 83", "ID 1")]}
#: Every leg runs CENTRE TO CENTRE of the instances, not from the labels (Eric,
#: 2026-10-27). 2D instance centres are measured off the panels' own vector
#: keypoint dots (`_instance_centres`); the 3D render is raster, so its centre is
#: read off the render (fig1c mm, trunk centre). Both ID 1 legs converge on it, and
#: one arrow runs on to the centre of its reprojection.
ID1_3D = (108.5, 10.0)


def _rep_dots():
    """Every keypoint dot (centre mm, fill) in fig1c's cam 0 reprojection tile."""
    pg = fitz.open(FIGURES / "fig1" / "fig1a_reconstruction.pdf")[0]
    out = []
    for d in pg.get_drawings():
        f, r = d.get("fill"), d["rect"]
        if f is None or r.x1 / MM > 63:
            continue
        w = r.width / MM
        if 0.2 < w < 1.5 and abs(w - r.height / MM) < 0.2:
            out.append((((r.x0 + r.x1) / 2 / MM, (r.y0 + r.y1) / 2 / MM), f))
    return out


def _instance_centres(pdf):
    """[(centre_mm, fill_rgb)] per skeleton in a panel PDF: its keypoint dots grouped
    by fill colour within a tile, and the centre taken as the mean of the 11 dots
    nearest their median -- i.e. the body, with the 4 tail nodes left out, since a
    plain mean of 15 nodes is dragged toward the tail."""
    import numpy as np
    pg = fitz.open(pdf)[0]
    groups = {}
    for d in pg.get_drawings():
        f, r = d.get("fill"), d["rect"]
        if f is None:
            continue
        w, h = r.width / MM, r.height / MM
        if 0.2 < w < 1.5 and abs(w - h) < 0.2:
            c = ((r.x0 + r.x1) / 2 / MM, (r.y0 + r.y1) / 2 / MM)
            # one group per colour per ~45 mm tile (same colour recurs across tiles)
            groups.setdefault((tuple(round(v, 3) for v in f), int(c[0] // 45)), []).append(c)
    out = []
    for (f, _), pts in groups.items():
        a = np.array(pts)
        m = np.median(a, 0)
        keep = a[np.argsort(((a - m) ** 2).sum(1))[:max(3, len(a) - 4)]]
        out.append((tuple(keep.mean(0)), f))
    return out


def _nearest(centres, xy, x_range):
    cands = [c for c in centres if x_range[0] <= c[0][0] <= x_range[1]]
    return min(cands, key=lambda c: (c[0][0] - xy[0]) ** 2 + (c[0][1] - xy[1]) ** 2)


#: the 3D render's own top-edge colour (measured, 300 dpi): the strip a downward
#: shift exposes above the image is filled with it, so it reads as render
RENDER_TOP = (253.8 / 255, 253.9 / 255, 251.7 / 255)


def _fig1_3d_shift(t, th1, th3, xs, rows_y, tiles_a, ty, tri3d):
    """How far (fig1c mm) to move the 3D render DOWN in its tile so the cam 7 ID 1
    leg can run STRAIGHT to the 3D centre without crossing cam 0's "cam 0 mid"
    badge (Eric, 2026-10-27: "move it down ... and crop a bit" instead of the bend).
    Smallest shift that clears the badge by 0.8 mm, plus 0.5 mm."""
    sx = t / (tiles_a[3][1] - tiles_a[3][0])
    cen = _instance_centres(FIGURES / "fig1" / "fig1a_tracking.pdf")
    spans = [([v / MM for v in sp["bbox"]], sp["text"].strip())
             for b in fitz.open(FIGURES / "fig1" / "fig1a_tracking.pdf")[0].get_text("dict")["blocks"]
             for ln in b.get("lines", []) for sp in ln["spans"]]
    ib = next(bb for bb, tx in spans if tx == "ID 1" and tiles_a[3][0] <= bb[0] <= tiles_a[3][1])
    q7, _ = _nearest(cen, ((ib[0] + ib[2]) / 2, ib[3] + 4), tiles_a[3])
    p7 = (xs[1] + (q7[0] - tiles_a[3][0]) * sx, rows_y[1] + (q7[1] - ty[0]) * sx)
    bb = next(bb for bb, tx in spans if tx == "cam 0 mid" and tiles_a[2][0] <= bb[0] <= tiles_a[2][1])
    bx0, by0 = xs[0] + (bb[0] - tiles_a[2][0]) * sx - 0.8, rows_y[1] + (bb[1] - ty[0]) * sx - 0.8
    bx1, by1 = xs[0] + (bb[2] - tiles_a[2][0]) * sx + 0.8, rows_y[1] + (bb[3] - ty[0]) * sx + 0.8
    s3 = t / (tri3d[2] - tri3d[0])
    for i in range(0, 80):
        d = i * 0.25
        p3 = (xs[0] + (ID1_3D[0] - tri3d[0]) * s3, rows_y[2] + (ID1_3D[1] - tri3d[1] + d) * s3)
        hit = any(bx0 <= p7[0] + (p3[0] - p7[0]) * f <= bx1 and
                  by0 <= p7[1] + (p3[1] - p7[1]) * f <= by1
                  for f in [k / 400 for k in range(401)])
        if not hit:
            return d + 0.5
    raise RuntimeError("no shift clears the badge")


def _hex(c):
    return tuple(((c >> sh) & 255) / 255 for sh in (16, 8, 0))


def _fig1_flow(s, t, th1, th3, xs, rows_y, tiles_a, ty, tri3d, rep_zoom):
    import math
    MMp = MM
    spans = {}
    for b in fitz.open(FIGURES / "fig1" / "fig1a_tracking.pdf")[0].get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for sp in ln["spans"]:
                spans.setdefault(sp["text"].strip(), []).append(
                    ([v / MMp for v in sp["bbox"]], _hex(sp["color"])))

    def tile_map(clip, dx, dy, dw, dh):
        sx = dw / (clip[2] - clip[0])
        return lambda x, y: (dx + (x - clip[0]) * sx, dy + (y - clip[1]) * sx)

    maps = {(r, c): tile_map((tiles_a[2 * r + c][0], ty[0], tiles_a[2 * r + c][1], ty[1]),
                             xs[c], rows_y[r], t, th1) for r in (0, 1) for c in (0, 1)}

    def span_in(text, tile):
        x0 = tiles_a[tile][0]
        for bb, col in spans[text]:
            if tiles_a[tile][0] <= bb[0] <= tiles_a[tile][1]:
                return bb, col
        raise KeyError(text)

    pg = s.page
    P = lambda q: fitz.Point(q[0] * MMp, q[1] * MMp)  # noqa: E731

    def halo(p0, p1):
        pg.draw_line(P(p0), P(p1), color=(1, 1, 1), width=(FLOW_LW + 0.9) * MMp,
                     lineCap=1, stroke_opacity=0.85)

    def head(p0, p1, col, size=HEAD):
        import math
        a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        tip = p1
        l = (p1[0] - size * math.cos(a - 0.45), p1[1] - size * math.sin(a - 0.45))
        r = (p1[0] - size * math.cos(a + 0.45), p1[1] - size * math.sin(a + 0.45))
        pg.draw_polyline([P(l), P(tip), P(r), P(l)], color=(1, 1, 1), fill=col,
                         width=0.25 * MMp, closePath=True)

    def shorten(p0, p1, by):
        import math
        L = math.dist(p0, p1)
        return (p1[0] - (p1[0] - p0[0]) * by / L, p1[1] - (p1[1] - p0[1]) * by / L)

    def gradient(p0, p1, c0, c1, arrow=True, head_at=None):
        """`head_at` = a point ON the leg where the head's tip goes (the leg runs on
        past it to p1); default is the end."""
        import math
        if head_at is not None:
            halo(p0, p1)
            gradient(p0, p1, c0, c1, arrow=False)
            f = math.dist(p0, head_at) / math.dist(p0, p1)
            head(p0, head_at, tuple((1 - f) * a + f * b for a, b in zip(c0, c1)))
            return
        end = shorten(p0, p1, HEAD * 0.75) if arrow else p1
        halo(p0, end)
        L = math.dist(p0, end)
        n = max(2, int(L / (DASH + GAPL)))
        for i in range(n):
            f0, f1 = i / n, i / n + DASH / (DASH + GAPL) / n
            m = (f0 + f1) / 2
            col = tuple((1 - m) * a + m * b for a, b in zip(c0, c1))
            q0 = (p0[0] + (end[0] - p0[0]) * f0, p0[1] + (end[1] - p0[1]) * f0)
            q1 = (p0[0] + (end[0] - p0[0]) * f1, p0[1] + (end[1] - p0[1]) * f1)
            pg.draw_line(P(q0), P(q1), color=col, width=FLOW_LW * MMp, lineCap=1)
        if arrow:
            head(p0, p1, c1)

    def dotted(p0, p1, col, arrow=False):
        """Round dots on the white halo -- "dotted = reprojected", as in Fig 3a."""
        import math
        end = shorten(p0, p1, HEAD * 0.75) if arrow else p1
        halo(p0, end)
        L = math.dist(p0, end)
        n = max(2, int(L / DOT_STEP))
        for i in range(n + 1):
            f = i / n
            pg.draw_circle(P((p0[0] + (end[0] - p0[0]) * f, p0[1] + (end[1] - p0[1]) * f)),
                           FLOW_LW * 0.55 * MMp, color=None, fill=col)
        if arrow:
            head(p0, p1, col)

    def solid(p0, p1, col, arrow=False):
        end = shorten(p0, p1, HEAD * 0.75) if arrow else p1
        halo(p0, end)
        pg.draw_line(P(p0), P(end), color=col, width=FLOW_LW * MMp, lineCap=1)
        if arrow:
            head(p0, p1, col)

    def dot(q, col, r=0.8):
        pg.draw_circle(P(q), r * MMp, color=(1, 1, 1), fill=col, width=0.3 * MMp)

    # every label chip and keypoint dot in each row-2 tile, in sheet mm, for
    # keeping the track -> ID heads off the instances they point at
    pg_a = fitz.open(FIGURES / "fig1" / "fig1a_tracking.pdf")[0]
    kp_a = []
    for d in pg_a.get_drawings():
        f, r = d.get("fill"), d["rect"]
        if f is not None and 0.2 < r.width / MMp < 1.5 and abs(r.width - r.height) / MMp < 0.2:
            kp_a.append(((r.x0 + r.x1) / 2 / MMp, (r.y0 + r.y1) / 2 / MMp, r.width / MMp / 2))

    def head_clear(tip, p0, tile, cl=CLEAR):
        import numpy as np
        m = maps[(1, tile - 2)]
        lo, hi = tiles_a[tile]
        base = shorten(p0, tip, HEAD)
        hh = HEAD * math.sin(0.45)
        a, b = np.array(base), np.array(tip)

        def d_seg(q):
            q = np.array(q)
            u = np.clip(np.dot(q - a, b - a) / np.dot(b - a, b - a), 0, 1)
            return np.linalg.norm(q - (a + u * (b - a)))
        for x, y_, r in kp_a:
            if lo <= x <= hi and d_seg(m(x, y_)) < hh + r * (m(1, 0)[0] - m(0, 0)[0]) + 0.8:
                return False
        for txt, lst in spans.items():
            for bb, _ in lst:
                if lo <= bb[0] <= hi and txt.startswith("ID"):
                    x0, y0 = m(bb[0] - CHIP_PAD, bb[1] - CHIP_PAD)
                    x1, y1 = m(bb[2] + CHIP_PAD, bb[3] + CHIP_PAD)
                    if (x0 - hh - cl <= tip[0] <= x1 + hh + cl
                            and y0 - cl <= tip[1] and base[1] <= y1 + cl):
                        return False
        return True

    cen_a = _instance_centres(FIGURES / "fig1" / "fig1a_tracking.pdf")
    # 1. track instance -> identity instance, same camera; gradient track -> ID colour
    id1_pts, dots = [], []
    for c, pairs in TRACK_TO_ID.items():
        for tr, idl in pairs:
            tb, _ = span_in(tr, c)                   # row 1 tile = tiles_a[c]
            ib, _ = span_in(idl, 2 + c)              # row 2 tile = tiles_a[2 + c]
            # the instance each label names: nearest skeleton below it in its tile
            (q0, tcol) = _nearest(cen_a, ((tb[0] + tb[2]) / 2, tb[3] + 4), tiles_a[c])
            (q1, icol) = _nearest(cen_a, ((ib[0] + ib[2]) / 2, ib[3] + 4), tiles_a[2 + c])
            p0, p1 = maps[(0, c)](*q0), maps[(1, c)](*q1)
            # the ID label sits right above the instance, so a centre-to-centre leg
            # runs into its chip. Slide the vertical leg left just far enough that
            # its halo clears the chip; its dots move with it and stay on the body.
            # The leg STOPS at its head, which points down at the ID instance
            # without touching it (Eric, 2026-10-28: "get rid of the link between
            # the arrow head and the instance"). The tip walks down from the top of
            # the chips for as long as the head clears every chip and keypoint.
            # To let it get CLOSER (Eric: "a little bit more"), the leg moves left
            # just far enough that the head passes beside the ID chip with
            # HEAD_CLEAR to spare -- when the gap to the next chip fits the head.
            chip_l = maps[(1, c)](ib[0] - CHIP_PAD, ib[1])[0]
            xl = min(p0[0], chip_l - (FLOW_LW + 0.9) / 2 - CLEAR)
            hh = HEAD * math.sin(0.45)
            m1 = maps[(1, c)]
            lefts = [m1(bb[2] + CHIP_PAD, 0)[0] for tx, lst in spans.items() if tx.startswith("ID")
                     for bb, _ in lst if tiles_a[2 + c][0] <= bb[0] <= tiles_a[2 + c][1]
                     and bb[2] < ib[0] and bb[1] < ib[3] and bb[3] > ib[1]]
            if not lefts or chip_l - max(lefts) >= 2 * (hh + HEAD_CLEAR):
                xl = min(xl, chip_l - hh - HEAD_CLEAR - 0.05)
            p0, p1 = (xl, p0[1]), (xl, p1[1])
            chip_t = maps[(1, c)](0, ib[1] - CHIP_PAD)[1]
            tip = (xl, chip_t - CLEAR)
            y = tip[1]
            while y < p1[1]:
                y += 0.1
                if not head_clear((xl, y), p0, 2 + c, HEAD_CLEAR):
                    break
                tip = (xl, y)
            gradient(p0, tip, tcol, icol)
            dots += [(p0, tcol), (p1, icol)]
            id1_pts.append((c, p1, icol))
    # 2. both ID 1 labels -> the green 3D animal, solid ID colour
    m3 = tile_map(tri3d, xs[0], rows_y[2], t, th3)
    p3d = m3(*ID1_3D)
    for c, p0, col in id1_pts:
        solid(p0, p3d, col)       # straight: the 3D tile is shifted to make room
    # 3. the 3D animal -> its reprojection in cam 0, solid ID colour with a head
    mr = tile_map(rep_zoom, xs[1], rows_y[2], t, th3)
    green = id1_pts[0][2]
    rep_c, _ = _nearest(_instance_centres(FIGURES / "fig1" / "fig1a_reconstruction.pdf"),
                        (25.6, 9.6), (0.08, 62.87))
    # DOTTED (a reprojection), STRAIGHT and HORIZONTAL from the 3D pose to the
    # GREEN (ID 1) reprojection (Eric, 2026-10-28: "make it straight horizontal and
    # collide with the yellow orange instance" ... "needs to go to the GREEN
    # instance, pull it over to the right"). The orange (ID 2) reprojection sits in
    # front of the green one, so the leg runs through it, and the tip lands on the
    # first green keypoint the line meets.
    import numpy as np
    gfill = min({f for _, f in _rep_dots()},
                key=lambda f: sum((a - b) ** 2 for a, b in zip(f, green)))
    G = np.array([mr(*q) for q, f in _rep_dots() if f == gfill
                  and rep_zoom[0] <= q[0] <= rep_zoom[2] and rep_zoom[1] <= q[1] <= rep_zoom[3]])
    y = p3d[1]
    on_line = G[np.abs(G[:, 1] - y) < 1.2]
    tip = (on_line[:, 0].min(), y)
    dotted(p3d, tip, green, arrow=True)
    for q, col in dots:
        dot(q, col)
    dot(p3d, green, 0.9)


# ---------------------------------------------------------------------------
def fig1():
    d = FIGURES / "fig1"
    trk, rec = d / "fig1a_tracking.pdf", d / "fig1a_reconstruction.pdf"
    # tile boxes (mm) measured from the panel PDFs' own image placements
    t_y0, t_y1 = 5.24, 38.56
    tiles_a = [(0.08, 44.70), (45.17, 89.79), (90.21, 134.83), (135.30, 179.92)]
    r_y0, r_y1 = 0.71, 34.28
    cam0_rep = (0.08, r_y0, 62.87, r_y1)
    ar3 = (cam0_rep[2] - cam0_rep[0]) / (r_y1 - r_y0)
    # the 3D render tile [63.50, 128.90] is a touch wider than the reprojection
    # tile's aspect; crop it from the RIGHT (empty floor past the green tail), so
    # the animals sit as far right as the render allows -- that steepens nothing
    # and lets the cam 7 leg clear the "cam 0 mid" badge with less downward shift,
    # which keeps the blue mouse's feet (render y 29.65) in frame (Eric, 2026-10-27)
    R3D = (63.50, r_y0, 128.90, r_y1)              # the render image's own box
    DX3D = 1.5                                      # extra right shift, filled
    tri3d = (R3D[0] - DX3D, r_y0, R3D[0] - DX3D + ar3 * (r_y1 - r_y0), r_y1)
    ar1 = (tiles_a[0][1] - tiles_a[0][0]) / (t_y1 - t_y0)
    g = 0.8                                       # mm between tiles
    ls = 4.5                                      # rotated row-label strip
    pipe, rend = d / "fig1c_pipeline.pdf", d / "fig1b_render.pdf"
    # the two arena renders, each with its name, cut out of the render row
    arenas = [(0.51, 0.11, 88.82, 91.8), (91.19, 0.11, 179.45, 91.8)]
    ara = (arenas[0][2] - arenas[0][0]) / (arenas[0][3] - arenas[0][1])
    agap = 2.0
    # a = ROWS ARE STAGES (Eric, 2026-10-27: "tracks on the first row then ids on the
    # second row then the 3D triangulation"; anchor diagram dropped), columns are
    # cam 0 mid / cam 7 sideR -- row 3 is the cam 0 reprojection and the 3D render.
    # Two cameras wide leaves a third of the page, so b = the arena renders stacked
    # in that column at a's height:
    #   ls + 2t + g + GUTTER + w = PAGE_W,   2t/ar1 + t/ar3 + 2g = 2w/ara + agap
    c1 = 2 / ar1 + 1 / ar3
    t = (PAGE_W - ls - g - GUTTER + ara * (2 * g - agap) / 2) / (2 + ara * c1 / 2)
    w = PAGE_W - ls - g - GUTTER - 2 * t
    th1, th3 = t / ar1, t / ar3
    grid_h = 2 * th1 + th3 + 2 * g
    ph = pdf_size(pipe)[1]
    page_h = MARGIN + LETTER_LEAD + grid_h + LEAD + ph + MARGIN
    s = Sheet(page_h)

    y = MARGIN + LETTER_LEAD
    s.letter("a", 0, y, TITLES[(1, "a")])
    xs = [ls, ls + t + g]
    rows_y = [y, y + th1 + g, y + 2 * (th1 + g)]
    for row, (cam0, cam7) in enumerate([(tiles_a[0], tiles_a[1]), (tiles_a[2], tiles_a[3])]):
        s.pdf(trk, xs[0], rows_y[row], t, th1, clip=(cam0[0], t_y0, cam0[1], t_y1))
        s.pdf(trk, xs[1], rows_y[row], t, th1, clip=(cam7[0], t_y0, cam7[1], t_y1))
    # 3D render first, then its reprojection into cam 0 (Eric, 2026-10-27), the
    # reprojection ZOOMED (~2.2x) onto the ID 1 / ID 2 pair and labelled as such.
    # Its own "cam 0 mid" badge falls outside the zoom and is replaced below.
    rep_zoom = (9.15, 3.9, 37.65, 3.9 + 28.5 / ar3)
    d3 = _fig1_3d_shift(t, th1, th3, xs, rows_y, tiles_a, (t_y0, t_y1), tri3d)
    if r_y1 - d3 < 29.65 + 0.4:
        print(f"    WARNING: 3D shift {d3:.2f} mm crops the blue mouse (feet at 29.65)")
    tri3d = (tri3d[0], tri3d[1] - d3, tri3d[2], tri3d[3] - d3)   # content moves down
    # PyMuPDF trims a clip to the source page and re-fits what is left, so a clip
    # reaching above the page does NOT shift the content. Place the in-page part
    # lower instead, and fill the exposed strip with the render's own top colour.
    s3 = t / (tri3d[2] - tri3d[0])
    s.page.draw_rect(Sheet.R(xs[0], rows_y[2], t, th3), color=None, fill=RENDER_TOP)
    v = (max(tri3d[0], R3D[0]), max(tri3d[1], R3D[1]),
         min(tri3d[2], R3D[2]), min(tri3d[3], R3D[3]))   # the part that is render
    s.pdf(rec, xs[0] + (v[0] - tri3d[0]) * s3, rows_y[2] + (v[1] - tri3d[1]) * s3,
          (v[2] - v[0]) * s3, (v[3] - v[1]) * s3, clip=v)
    s.pdf(rec, xs[1], rows_y[2], t, th3, clip=rep_zoom)
    hb = fitz.Font("Helvetica-Bold")
    if d3 > 0:   # the render's own "triangulated 3D" label is cropped off: redraw it
        lab = "triangulated 3D"
        s.page.insert_text(fitz.Point((xs[0] + t - 1.2 - hb.text_length(lab, fontsize=6.5) / MM) * MM,
                                      (rows_y[2] + th3 - 1.3) * MM), lab,
                           fontname="VarHeboBold", fontsize=6.5, color=INK)
    lab = "cam 0 mid · reprojection"
    s.page.insert_text(fitz.Point((xs[1] + t - 1.2 - hb.text_length(lab, fontsize=6.5) / MM) * MM,
                                  (rows_y[2] + th3 - 1.3) * MM), lab,
                       fontname="VarHeboBold", fontsize=6.5, color=(1, 1, 1))
    _fig1_flow(s, t, th1, th3, xs, rows_y, tiles_a, (t_y0, t_y1), tri3d, rep_zoom)
    # row labels, rotated, centred on each row
    helv = fitz.Font("Helvetica")
    for ry, rh, lab in zip(rows_y, [th1, th1, th3],
                           ["per-camera tracks", "LUC3D identities", "triangulated 3D"]):
        tl = helv.text_length(lab, fontsize=7.0) / MM
        s.page.insert_text(fitz.Point((ls - 1.4) * MM, (ry + rh / 2 + tl / 2) * MM), lab,
                           fontname="VarHelv", fontsize=7.0, color=(0.35, 0.35, 0.35),
                           rotate=90)

    bx = ls + 2 * t + g + GUTTER
    ah = w / ara
    for i, clip in enumerate(arenas):
        s.pdf(rend, bx, y + i * (ah + agap), w, ah, clip=clip)
    s.letter("b", bx, y, TITLES[(1, "b")])

    y += grid_h + LEAD
    s.pdf(pipe, 0, y, PAGE_W)
    s.letter("c", 0, y, TITLES[(1, "c")])
    s.save("fig1")
    print(f"    a block {ls + 2 * t + g:.1f} x {grid_h:.1f} mm (tiles {t:.1f} wide); "
          f"arena column {w:.1f} mm; 3D render moved down {d3:.2f} mm (fig1c units)")


# ---------------------------------------------------------------------------
def fig6():
    """Supp Fig 4 (repo fig6) without the anchor diagram, which moved to Fig 2a. Lettered in reading order: the a-d tracker block beside the
    s-DANNCE survival (e, was g), then the staleness sweep (f, was f). Both are
    cited in that order in the tex (Methods datasets, then the anchor section)."""
    d = FIGURES / "fig6"
    block = d / "fig6a_block.pdf"
    sd = d / "fig6e_sdannce_survival.pdf"
    st = d / "fig6f_switch_reduction.pdf"
    bw, bh = pdf_size(block)
    sw, sh_ = pdf_size(sd)
    tw, th = pdf_size(st)
    sdw = sw * bh / sh_                       # s-DANNCE at the block's height
    row1 = bw + GUTTER + sdw
    page_h = MARGIN + LETTER_LEAD + bh + LEAD + th + MARGIN
    s = Sheet(page_h)
    y = MARGIN + LETTER_LEAD
    x0 = (PAGE_W - row1) / 2
    s.pdf(block, x0, y, bw)
    s.letter("a", x0, y, TITLES[(6, "a")])
    # the block's sub-letters, as assemble.EXTRA_LETTERS draws them (titles flush)
    for sub, dx, dy in [("b", 43.51, 0.0), ("c", 0.0, 34.95), ("d", 43.51, 34.95)]:
        sx = x0 + dx + (0.6 if dx == 0.0 else 0.0)
        s.page.insert_text(fitz.Point(sx * MM, (y + dy - 1.2) * MM), sub,
                           fontname="VarHeboBold", fontsize=LETTER_PT, color=INK)
        s.page.insert_text(fitz.Point((sx + 4.4) * MM, (y + dy - 1.2) * MM),
                           TITLES[(6, sub)], fontname="VarHeboBold",
                           fontsize=TITLE_PT, color=INK)
    s.pdf(sd, x0 + bw + GUTTER, y, sdw)
    s.letter("e", x0 + bw + GUTTER, y, TITLES[(6, "e")], sd, sdw)
    y += bh + LEAD
    xs = (PAGE_W - tw) / 2
    s.pdf(st, xs, y, tw)
    s.letter("f", xs, y, TITLES[(6, "f")], st, tw)
    s.save("fig6")


# ---------------------------------------------------------------------------
def fig3():
    """Fig 3, the labeling figure, row for row as assemble.LAYOUTS[3] places it.
    a and d carry no "anchor" (panels/fig3_01_protocol.py, fig3_04_baseline_angle.py
    built with FIG_VARIANT=x7), since Fig 2 uses the word for the tracker's 3D anchor."""
    d = FIGURES / "fig3"
    prot = d / "fig3a_protocol.pdf"
    ang = d / "fig3d_baseline_angle.pdf"
    titles = {l: TITLES[(3, l)] for l in "abcdefg"}
    rows = [[("a", prot)],
            [(l, d / f"fig3{l}_{s_}.pdf") for l, s_ in
             (("b", "placements_vs_rig"), ("c", "reprojection_accuracy"))] + [("d", ang)],
            [(l, d / f"fig3{l}_{s_}.pdf") for l, s_ in
             (("e", "solver_accuracy"), ("f", "per_session"), ("g", "time_per_keypoint"))]]
    sizes = [[pdf_size(p_) for _, p_ in r] for r in rows]
    page_h = MARGIN + sum(LETTER_LEAD + max(h for _, h in sz) for sz in sizes) \
        + ROW_GAP * (len(rows) - 1) + MARGIN
    s = Sheet(page_h)
    y = MARGIN
    for r, sz in zip(rows, sizes):
        y += LETTER_LEAD
        x = (PAGE_W - (sum(w for w, _ in sz) + GUTTER * (len(sz) - 1))) / 2
        for (l, p_), (w, h) in zip(r, sz):
            s.pdf(p_, x, y, w)
            s.letter(l, x, y, titles[l], None if l in ("a", "e") else p_, w)
            x += w + GUTTER
        y += max(h for _, h in sz) + ROW_GAP
    s.save("fig3")


BUILDERS = {1: fig1, 2: fig2, 3: fig3, 6: fig6}

if __name__ == "__main__":
    for n in [int(a) for a in sys.argv[1:] if a.isdigit()] or sorted(BUILDERS):
        print(f"{BUILDERS[n].__name__}:")
        BUILDERS[n]()
