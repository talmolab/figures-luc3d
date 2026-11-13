/**
 * Fig 2G cross-check (md56, 2026-10-14): do two LUC3D code versions solve the SAME 3D
 * points, and how far apart are they? Loads triangulation.js from POSE_A and POSE_B in one
 * process, feeds both identical undistorted inputs (undistorted once, with version B's
 * camera model), and reports the max |dX| for the DLT and the refined (BA) solve over the
 * first --n keypoints of calibration 0. Timing is NOT measured here (two module instances in
 * one process would share JIT state); use fig2_solvers_time_luc3d.mjs per version for that.
 *   POSE_A=<dir> POSE_B=<dir> FIG2_SUFFIX=md56 node figs/fig2g_ab_versions.mjs --n 200000
 */
import { register } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const SFX = process.env.FIG2_SUFFIX ? `_${process.env.FIG2_SUFFIX}` : '';
register(pathToFileURL(path.join(HERE, 'fig2_solvers_hooks.mjs')).href, import.meta.url);
const load = async d => ({
    tri: await import(pathToFileURL(path.join(d, 'triangulation.js')).href),
    pd: await import(pathToFileURL(path.join(d, 'pose-data.js')).href)});
const A = await load(path.resolve(process.env.POSE_A)), B = await load(path.resolve(process.env.POSE_B));
const i = process.argv.indexOf('--n'); const N = i >= 0 ? +process.argv[i + 1] : 200000;
const data = JSON.parse(fs.readFileSync(path.join(HERE, 'out', `fig4_input${SFX}.json`), 'utf8'));
const buf = fs.readFileSync(path.join(HERE, 'out', data.bin));
const C = data.n_cameras;
const OBS = new Float64Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), 0, data.keypoints * C * 2);
const cal = data.calibrations[0];
const mk = pd => cal.map(c => new pd.Camera(c.name, c.matrix, c.distortions, c.rvec, c.tvec, c.size));
const camA = mk(A.pd), camB = mk(B.pd);
const PsA = camA.map(c => c.projectionMatrix), PsB = camB.map(c => c.projectionMatrix);
const n = Math.min(N, data.blocks[0].count);
let dP = 0; for (let c = 0; c < C; c++) for (let j = 0; j < 12; j++) dP = Math.max(dP, Math.abs(PsA[c].flat()[j] - PsB[c].flat()[j]));
const D = []; let dlt = 0, ba = 0, und = 0, nA = 0, nB = 0, solved = 0;
for (let k = 0; k < n; k++) {
    const raw = []; for (let c = 0; c < C; c++) { const q = (k * C + c) * 2;
        raw.push(Number.isFinite(OBS[q]) && Number.isFinite(OBS[q + 1]) ? [OBS[q], OBS[q + 1]] : null); }
    const uB = raw.map((p, c) => (p ? camB[c].undistortPoint(p) : null));
    const uA = raw.map((p, c) => (p ? camA[c].undistortPoint(p) : null));
    for (let c = 0; c < C; c++) if (uA[c]) und = Math.max(und, Math.abs(uA[c][0] - uB[c][0]), Math.abs(uA[c][1] - uB[c][1]));
    const xa = A.tri.triangulatePointDLT(uB, PsA), xb = B.tri.triangulatePointDLT(uB, PsB);
    if (!xa) nA++; if (!xb) nB++; if (!xa || !xb) continue;
    solved++;
    let dk = 0; for (let j = 0; j < 3; j++) dk = Math.max(dk, Math.abs(xa[j] - xb[j]));
    dlt = Math.max(dlt, dk); D.push(dk);
    const ra = A.tri.triangulatePointBA(raw, PsA, xb, { cameras: camA }), rb = B.tri.triangulatePointBA(raw, PsB, xb, { cameras: camB });
    if (ra && rb) for (let j = 0; j < 3; j++) ba = Math.max(ba, Math.abs(ra[j] - rb[j]));
}
D.sort((a, b) => a - b); const q = f => D[Math.min(D.length - 1, Math.floor(f * D.length))];
const cnt = t => D.filter(x => x > t).length;
console.log(JSON.stringify({ dlt_dX_quantiles: { p50: q(0.5), p90: q(0.9), p99: q(0.99), p999: q(0.999) },
    n_dlt_dX_over: { '1e-9': cnt(1e-9), '1e-3': cnt(1e-3), '0.1': cnt(0.1), '1': cnt(1) } }));
console.log(JSON.stringify({ n, solved, nullA: nA, nullB: nB, max_abs_dP: dP, max_abs_dUndistort_px: und,
    max_abs_dX_dlt: dlt, max_abs_dX_ba_same_seed: ba, units: 'calibration units (~mm)',
    mm_per_unit_block0: data.blocks[0].mm_per_unit }));
