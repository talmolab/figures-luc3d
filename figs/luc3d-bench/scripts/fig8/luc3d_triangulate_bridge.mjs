// Fig 8 bridge: run LUCID's SHIPPED triangulators from Node on arbitrary
// (observations, projection matrices) batches, so the comparison against
// Lightning Pose 3D's triangulator runs the real app code and not a
// reimplementation of it.
//
// pose/triangulation.js is an ES module whose imports are all app-level
// (pose-data, ui/*, import-export/*); triangulatePointDLT / triangulatePointBA /
// reprojectPoint need none of them. So the imports are stripped and the file is
// evaluated in a vm sandbox — the same pattern scripts/luc3d_track_all.mjs uses.
//
// stdin  : {"cases": [{"obs": [[x,y]|null, ...], "P": [[[..4],[..4],[..4]]|null, ...]}]}
// stdout : {"dlt": [[x,y,z]|null], "ba": [[x,y,z]|null]}
import vm from 'node:vm';
import fs from 'node:fs';
import path from 'node:path';

const LUCID = process.env.LUCID_DIR
    || '/root/vast/eric/sleap-3d-gui/scratch/repos/lucid';

// pose-data.js too: the lucid-bundle-adj solver's native-space mode needs real
// Camera objects (it calls camera.distortPoint() and camera.projectionMatrix).
const src = [
    fs.readFileSync(path.join(LUCID, 'pose', 'pose-data.js'), 'utf8'),
    fs.readFileSync(path.join(LUCID, 'pose', 'triangulation.js'), 'utf8'),
].join('\n');

// Strip module syntax: the import lines (app-level deps we do not need) and the
// `export` keyword (vm has no module scope). Nothing else is touched.
const stripped = src
    .replace(/^import[\s\S]*?from\s+['"][^'"]+['"];?\s*$/gm, '')
    .replace(/^export\s+(async\s+)?(function|const|let|var|class)\s/gm, '$1$2 ');

const sandbox = {
    console, Math, JSON, Number, String, Boolean, Array, Object, Map, Set,
    Error, TypeError, RangeError, parseInt, parseFloat, isNaN, isFinite,
    NaN, Infinity, Float64Array, Float32Array, Uint8Array, ArrayBuffer,
    Date, RegExp, Promise,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(stripped, sandbox, { filename: 'pose/lucid-combined.js' });
vm.runInContext('globalThis.Camera = Camera;', sandbox);

const dlt = sandbox.triangulatePointDLT;
const ba = sandbox.triangulatePointBA;
// The BATCHED entry points -- what the app actually calls: one call per frame with
// every keypoint, not one call per point. Fig 4f's "per keypoint" cost should come
// from these, because per-call overhead amortised over a frame is the cost a user
// actually pays. (Internally they are still JS loops over the per-point solver, so
// this measures amortisation, not vectorisation.)
const dltMany = sandbox.triangulatePoints;
const baMany = sandbox.triangulatePointsBA;
if (typeof dlt !== 'function' || typeof ba !== 'function') {
    throw new Error('triangulation.js did not expose triangulatePointDLT/triangulatePointBA');
}

let raw = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (d) => { raw += d; });
process.stdin.on('end', () => {
    const { cases, refine, bench, repeats, camDefs } = JSON.parse(raw);

    // Native-space refinement (lucid-bundle-adj): observations stay DISTORTED and
    // the solver is handed Camera objects so its residual lives in each camera's
    // own pixels. The DLT seed still comes from undistorted coords, exactly as
    // figs/fig4_measure.mjs does it -- a DLT on distorted points would be biased.
    const cameras = camDefs
        ? camDefs.map(c => new sandbox.Camera(c.name, c.matrix, c.dist, c.rvec, c.tvec, c.size))
        : null;
    const baOpts = cameras ? { cameras } : {};
    const obsFor = (c) => (cameras && c.raw ? c.raw : c.obs);
    const outDlt = [];
    const outBa = [];

    // Results pass (untimed).
    const seeds = [];
    for (const c of cases) {
        const p = dlt(c.obs, c.P);
        outDlt.push(p);
        seeds.push(p);
        if (refine && p) {
            let q = null;
            try { q = ba(obsFor(c), c.P, p, baOpts); } catch (e) { q = null; }
            outBa.push(q && q.point ? q.point : (Array.isArray(q) ? q : null));
        } else {
            outBa.push(null);
        }
    }

    const payload = { dlt: outDlt, ba: outBa };

    if (bench) {
        // Time the two solvers in SEPARATE loops, after a warm-up pass that is not
        // counted. Interleaving them in one loop and timing from the first call
        // charges V8's JIT warm-up and whatever GC happens to fire to whichever
        // solver is running, which made an earlier version report the refinement as
        // 3x FASTER than the DLT it is seeded from -- impossible, and the tell that
        // the measurement was wrong rather than the result surprising.
        const warm = Math.min(cases.length, 2000);
        for (let i = 0; i < warm; i++) {
            const p = dlt(cases[i].obs, cases[i].P);
            if (refine && p) { try { ba(obsFor(cases[i]), cases[i].P, p, baOpts); } catch (e) {} }
        }
        const reps = repeats || 1;

        let tDlt = Infinity;
        for (let r = 0; r < reps; r++) {
            const t0 = performance.now();
            for (const c of cases) dlt(c.obs, c.P);
            tDlt = Math.min(tDlt, performance.now() - t0);
        }

        let tBa = Infinity;
        if (refine) {
            for (let r = 0; r < reps; r++) {
                const t0 = performance.now();
                for (let i = 0; i < cases.length; i++) {
                    const p = seeds[i];
                    if (!p) continue;
                    try { ba(obsFor(cases[i]), cases[i].P, p, baOpts); } catch (e) {}
                }
                tBa = Math.min(tBa, performance.now() - t0);
            }
        }
        // Batched entry points, timed the same way.
        let tDltB = Infinity, tBaB = Infinity;
        if (typeof dltMany === 'function') {
            const allObs = cases.map(c => obsFor(c));
            const Ps = cases[0].P;
            for (let r = 0; r < reps; r++) {
                const t0 = performance.now();
                dltMany(cases.map(c => c.obs), Ps);
                tDltB = Math.min(tDltB, performance.now() - t0);
            }
            if (refine && typeof baMany === 'function') {
                const seedFlat = dltMany(cases.map(c => c.obs), Ps);
                for (let r = 0; r < reps; r++) {
                    const t0 = performance.now();
                    baMany(allObs, Ps, seedFlat);
                    tBaB = Math.min(tBaB, performance.now() - t0);
                }
            }
        }

        const nSeed = seeds.filter(Boolean).length;
        payload.timing = {
            // best-of-repeats, warm; per keypoint
            dlt_us_per_keypoint: (tDlt * 1000) / cases.length,
            refine_only_us_per_keypoint: refine && nSeed ? (tBa * 1000) / nSeed : null,
            // what the app actually pays for a refined point: DLT seed + refinement
            refined_total_us_per_keypoint: refine && nSeed
                ? (tDlt * 1000) / cases.length + (tBa * 1000) / nSeed : null,
            dlt_batched_us_per_keypoint: isFinite(tDltB) ? (tDltB * 1000) / cases.length : null,
            refined_batched_total_us_per_keypoint: isFinite(tBaB)
                ? (tDltB * 1000) / cases.length + (tBaB * 1000) / cases.length : null,
            n: cases.length, n_seeded: nSeed, repeats: reps, warmup: warm,
        };
    }
    process.stdout.write(JSON.stringify(payload));
});
