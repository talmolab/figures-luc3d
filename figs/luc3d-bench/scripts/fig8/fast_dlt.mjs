// PROTOTYPE ONLY -- not wired into LUCID, and deliberately so.
//
// A batched, allocation-free DLT to see how much of LUC3D's 49.7 us/keypoint is the
// algorithm and how much is JS overhead. `pose/triangulation.js` is on every path in
// the app (tracker, Triangulate All, save), so it does not get rewritten on the
// strength of a benchmark; this measures the ceiling first.
//
// Where the current implementation spends its time, and what this changes:
//
//   1. It materialises A as an array-of-arrays: for C cameras that is 2C boxed
//      Float64 rows of length 4, allocated PER KEYPOINT and immediately garbage.
//      At 8.1M keypoints on the real project that is ~81M short-lived arrays.
//      -> here, A is never built. The normal matrix M = A^T A is accumulated
//         directly as 10 scalars (4x4 symmetric), which is also numerically the
//         same thing the SVD's null space is computed from.
//   2. It runs a Jacobi eigen-decomposition that itself allocates.
//      -> here, Jacobi runs on a fixed 16-element scratch Float64Array with the
//         rotations applied in place.
//   3. Every input point is a boxed [u,v] pair and every output a boxed [X,Y,Z].
//      -> here, input and output are flat Float64Arrays, the representation the
//         app already uses for Instance._xy and points3d (luc3d #185/#189).
//
// EQUIVALENCE IS NOT ASSUMED. Solving via M = A^T A squares the condition number
// relative to an SVD of A, so this is not guaranteed bit-identical; bench_fast_dlt.mjs
// reports the actual disagreement against the shipped solver on real data. If that
// disagreement is not negligible, this prototype is the wrong trade and should be
// abandoned rather than tuned.

/**
 * Batched DLT.
 * @param {Float64Array} obs   flat, C*N*2: obs[(c*N + n)*2 + {0,1}], NaN = missing
 * @param {Float64Array} P     flat, C*12: row-major 3x4 per camera
 * @param {number} N           keypoints
 * @param {number} C           cameras
 * @param {Float64Array} [out] flat 3N; allocated when omitted
 * @returns {Float64Array}     flat 3N, NaN where unsolvable
 */
export function triangulatePointsDLTFast(obs, P, N, C, out) {
    out = out || new Float64Array(3 * N);
    const M = new Float64Array(16);     // scratch: normal matrix, reused
    const V = new Float64Array(16);     // scratch: eigenvectors, reused

    for (let n = 0; n < N; n++) {
        M.fill(0);
        let used = 0;
        for (let c = 0; c < C; c++) {
            const o = (c * N + n) * 2;
            const u = obs[o], v = obs[o + 1];
            if (Number.isNaN(u) || Number.isNaN(v)) continue;
            used++;
            const p = c * 12;
            // r0 = u*P2 - P0 ; r1 = v*P2 - P1, accumulated straight into M
            const a0 = u * P[p + 8] - P[p + 0];
            const a1 = u * P[p + 9] - P[p + 1];
            const a2 = u * P[p + 10] - P[p + 2];
            const a3 = u * P[p + 11] - P[p + 3];
            const b0 = v * P[p + 8] - P[p + 4];
            const b1 = v * P[p + 9] - P[p + 5];
            const b2 = v * P[p + 10] - P[p + 6];
            const b3 = v * P[p + 11] - P[p + 7];
            M[0] += a0 * a0 + b0 * b0;
            M[1] += a0 * a1 + b0 * b1;
            M[2] += a0 * a2 + b0 * b2;
            M[3] += a0 * a3 + b0 * b3;
            M[5] += a1 * a1 + b1 * b1;
            M[6] += a1 * a2 + b1 * b2;
            M[7] += a1 * a3 + b1 * b3;
            M[10] += a2 * a2 + b2 * b2;
            M[11] += a2 * a3 + b2 * b3;
            M[15] += a3 * a3 + b3 * b3;
        }
        if (used < 2) { out[3 * n] = out[3 * n + 1] = out[3 * n + 2] = NaN; continue; }
        M[4] = M[1]; M[8] = M[2]; M[12] = M[3];
        M[9] = M[6]; M[13] = M[7]; M[14] = M[11];

        // Jacobi on the symmetric 4x4, in place, no allocation.
        V.fill(0); V[0] = V[5] = V[10] = V[15] = 1;
        for (let sweep = 0; sweep < 12; sweep++) {
            let off = 0;
            for (let i = 0; i < 4; i++) {
                for (let j = i + 1; j < 4; j++) off += M[i * 4 + j] * M[i * 4 + j];
            }
            if (off < 1e-22) break;
            for (let p2 = 0; p2 < 3; p2++) {
                for (let q = p2 + 1; q < 4; q++) {
                    const apq = M[p2 * 4 + q];
                    if (Math.abs(apq) < 1e-30) continue;
                    const app = M[p2 * 4 + p2], aqq = M[q * 4 + q];
                    const theta = (aqq - app) / (2 * apq);
                    const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                    const cs = 1 / Math.sqrt(t * t + 1), sn = t * cs;
                    for (let k = 0; k < 4; k++) {
                        const mkp = M[k * 4 + p2], mkq = M[k * 4 + q];
                        M[k * 4 + p2] = cs * mkp - sn * mkq;
                        M[k * 4 + q] = sn * mkp + cs * mkq;
                    }
                    for (let k = 0; k < 4; k++) {
                        const mpk = M[p2 * 4 + k], mqk = M[q * 4 + k];
                        M[p2 * 4 + k] = cs * mpk - sn * mqk;
                        M[q * 4 + k] = sn * mpk + cs * mqk;
                    }
                    for (let k = 0; k < 4; k++) {
                        const vkp = V[k * 4 + p2], vkq = V[k * 4 + q];
                        V[k * 4 + p2] = cs * vkp - sn * vkq;
                        V[k * 4 + q] = sn * vkp + cs * vkq;
                    }
                }
            }
        }
        let best = 0, bestVal = M[0];
        for (let i = 1; i < 4; i++) { if (M[i * 4 + i] < bestVal) { bestVal = M[i * 4 + i]; best = i; } }
        const w = V[3 * 4 + best];
        if (!(Math.abs(w) > 1e-12)) {
            out[3 * n] = out[3 * n + 1] = out[3 * n + 2] = NaN;
            continue;
        }
        out[3 * n] = V[0 * 4 + best] / w;
        out[3 * n + 1] = V[1 * 4 + best] / w;
        out[3 * n + 2] = V[2 * 4 + best] / w;
    }
    return out;
}
