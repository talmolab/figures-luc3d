// Benchmark + equivalence check for the batched DLT prototype against LUC3D's
// shipped triangulatePointDLT, on real data supplied as JSON on stdin.
//
// Reports BOTH numbers that matter, and the second is the one with a veto:
//   speed      us/keypoint for each, warm, best of N passes
//   agreement  max and median |fast - shipped| in mm over the same points
//
// Accumulating M = A^T A squares the condition number relative to an SVD of A, so a
// speedup here is only interesting if the disagreement is negligible against the
// ~2.9 px / few-mm scale the figures work at. If it is not, the prototype is the
// wrong trade and gets abandoned, not tuned.
import vm from 'node:vm';
import fs from 'node:fs';
import path from 'node:path';
import { triangulatePointsDLTFast } from './fast_dlt.mjs';

const LUCID = process.env.LUCID_DIR
    || '/root/vast/eric/sleap-3d-gui/scratch/repos/lucid';

function strip(src) {
    return src
        .replace(/^import[\s\S]*?from\s+['"][^'"]+['"];?\s*$/gm, '')
        .replace(/^export\s+(async\s+)?(function|const|let|var|class)\s/gm, '$1$2 ')
        .replace(/^export\s*\{[^}]*\}\s*;?\s*$/gm, '');
}

const sandbox = {
    console, Math, JSON, Number, String, Boolean, Array, Object, Map, Set,
    Error, TypeError, RangeError, parseInt, parseFloat, isNaN, isFinite,
    NaN, Infinity, Float64Array, Float32Array, Uint8Array, ArrayBuffer, Date, RegExp,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(strip(fs.readFileSync(path.join(LUCID, 'pose', 'triangulation.js'), 'utf8')),
                sandbox, { filename: 'triangulation.js' });
const dlt = sandbox.triangulatePointDLT;
const dltMany = sandbox.triangulatePoints;

let raw = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', d => { raw += d; });
process.stdin.on('end', () => {
    const { cases, repeats } = JSON.parse(raw);
    const N = cases.length;
    const C = cases[0].P.length;
    const reps = repeats || 3;

    // flat inputs for the prototype
    const obs = new Float64Array(C * N * 2);
    const P = new Float64Array(C * 12);
    for (let c = 0; c < C; c++) {
        const Pm = cases[0].P[c];
        for (let r = 0; r < 3; r++) for (let k = 0; k < 4; k++) P[c * 12 + r * 4 + k] = Pm[r][k];
    }
    for (let n = 0; n < N; n++) {
        for (let c = 0; c < C; c++) {
            const o = cases[n].obs[c];
            const i = (c * N + n) * 2;
            obs[i] = o ? o[0] : NaN;
            obs[i + 1] = o ? o[1] : NaN;
        }
    }

    // warm-up (both), untimed
    const warm = Math.min(N, 2000);
    for (let i = 0; i < warm; i++) dlt(cases[i].obs, cases[i].P);
    triangulatePointsDLTFast(obs, P, N, C);

    let tOld = Infinity;
    for (let r = 0; r < reps; r++) {
        const t0 = performance.now();
        for (let n = 0; n < N; n++) dlt(cases[n].obs, cases[n].P);
        tOld = Math.min(tOld, performance.now() - t0);
    }
    const out = new Float64Array(3 * N);
    let tNew = Infinity;
    for (let r = 0; r < reps; r++) {
        const t0 = performance.now();
        triangulatePointsDLTFast(obs, P, N, C, out);
        tNew = Math.min(tNew, performance.now() - t0);
    }

    // agreement against the shipped solver
    const diffs = [];
    let nBothNull = 0, nMismatchNull = 0;
    for (let n = 0; n < N; n++) {
        const a = dlt(cases[n].obs, cases[n].P);
        const bx = out[3 * n], by = out[3 * n + 1], bz = out[3 * n + 2];
        const bNull = Number.isNaN(bx);
        if (a == null && bNull) { nBothNull++; continue; }
        if ((a == null) !== bNull) { nMismatchNull++; continue; }
        diffs.push(Math.hypot(a[0] - bx, a[1] - by, a[2] - bz));
    }
    diffs.sort((x, y) => x - y);
    const q = f => (diffs.length ? diffs[Math.min(diffs.length - 1,
        Math.floor(f * diffs.length))] : NaN);

    process.stdout.write(JSON.stringify({
        n: N, cameras: C, repeats: reps,
        shipped_us_per_keypoint: (tOld * 1000) / N,
        fast_us_per_keypoint: (tNew * 1000) / N,
        speedup: tOld / tNew,
        agreement_mm: { median: q(0.5), p95: q(0.95), p99: q(0.99),
                        max: diffs.length ? diffs[diffs.length - 1] : NaN },
        n_compared: diffs.length, n_both_null: nBothNull, n_null_mismatch: nMismatchNull,
    }, null, 2));
});
