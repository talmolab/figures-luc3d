// Fig 8d: run LUC3D's SHIPPED cross-view tracker on social-DANNCE data.
//
// Loads pose/{pose-data,cross-view-tracker,triangulation,tracker}.js plus the real
// ui/settings.js into a node vm sandbox with module syntax stripped and the
// browser/UI surface stubbed, then calls runCrossViewTracker() -- the function
// trackAll() itself calls.
//
// DO NOT drive matchFrameInstances() here. It is the app's OLD per-frame matcher
// ("The app no longer uses it", pose/tracker.js:223); the temporal identity layer
// lives in the CrossViewTracker that createTrackerRun/stepTrackerFrame carry
// across frames. A first version of this script looped matchFrameInstances with
// prevAssignments/prevTargets3d and produced identity that flipped on ~44% of
// frames with a median run length of 2 -- and, tellingly, the SAME switch count
// (3998/9000) in three different sessions, because with no temporal state the
// identity index just followed the instance order the preparer had shuffled.
// Those numbers were an artefact of the harness. See figs/PLAN-fig8.md.
//
// Input : the JSON written by scripts/fig8/sdannce_prepare.py. Instance order
//         inside each camera-frame is already SHUFFLED there, so nothing about
//         the true identity leaks in through ordering.
// Output: per-frame {camera -> [assigned identity per instance]} plus the true
//         identity of each instance, for scripts/fig8/sdannce_score.py.
import vm from 'node:vm';
import fs from 'node:fs';
import path from 'node:path';

const LUCID = process.env.LUCID_DIR
    || '/root/vast/eric/sleap-3d-gui/scratch/repos/lucid';

function strip(src) {
    return src
        .replace(/^import[\s\S]*?from\s+['"][^'"]+['"];?\s*$/gm, '')
        .replace(/^export\s+(async\s+)?(function|const|let|var|class)\s/gm, '$1$2 ')
        .replace(/^export\s*\{[^}]*\}\s*;?\s*$/gm, '');
}

const noop = () => {};
const sandbox = {
    console, Math, JSON, Number, String, Boolean, Array, Object, Map, Set,
    WeakMap, WeakSet, Symbol, Error, TypeError, RangeError, Promise,
    parseInt, parseFloat, isNaN, isFinite, NaN, Infinity, Date, RegExp,
    Float64Array, Float32Array, Uint8Array, Int32Array, ArrayBuffer, DataView,
    setTimeout, clearTimeout,
    // --- browser surface the modules touch on load ---
    localStorage: { getItem: () => null, setItem: noop, removeItem: noop },
    document: { getElementById: () => null, querySelector: () => null,
                querySelectorAll: () => [], createElement: () => ({ style: {} }),
                addEventListener: noop },
    window: {}, navigator: { userAgent: 'node' },
    // --- app surface the pose modules import ---
    state: {}, timeline: {}, viewport3d: {}, interactionManager: {},
    getActiveSession: () => sandbox.__session || null,
    setStatus: noop, showLoading: noop, hideLoading: noop, markDirty: noop,
    drawAllOverlays: noop, updateInfoPanel: noop, updateTriangulationBadge: noop,
    setReprojErrorVisible: noop, update3DViewport: noop,
};
sandbox.globalThis = sandbox;
sandbox.window = sandbox;
vm.createContext(sandbox);

// settings.js FIRST so the tracker gets the app's real defaults (node weights,
// tracking thresholds, camera inclusion) rather than invented ones.
const FILES = [
    'ui/settings.js',
    'pose/pose-data.js',
    'pose/cross-view-tracker.js',
    'pose/triangulation.js',
    'pose/tracker.js',
];
const combined = FILES.map(f => `\n//#region ${f}\n` + strip(fs.readFileSync(path.join(LUCID, f), 'utf8'))).join('\n');
vm.runInContext(combined, sandbox, { filename: 'lucid-combined.js' });

// Script-level class/const declarations live in the global LEXICAL scope, which is
// shared across scripts in this realm but is not reachable as globalThis.<name>.
// Hoist the handful we drive directly.
vm.runInContext(`
    globalThis.Camera = Camera;
    globalThis.Instance = Instance;
    globalThis.FrameGroup = FrameGroup;
    globalThis.Session = Session;
    globalThis.Skeleton = Skeleton;
`, sandbox);

const { Camera, Instance, FrameGroup, Session, runCrossViewTracker } = sandbox;
if (typeof runCrossViewTracker !== 'function') {
    throw new Error('runCrossViewTracker not exposed by the sandbox');
}

// --------------------------------------------------------------------------- //
function parseToml(text, CameraCls) {
    const cams = [];
    let cur = null;
    for (const raw of text.split('\n')) {
        const line = raw.trim();
        if (!line) continue;
        const sec = line.match(/^\[(cam_\d+)\]$/);
        if (sec) { cur = {}; cams.push(cur); continue; }
        if (line.startsWith('[')) { cur = null; continue; }
        if (!cur) continue;
        const m = line.match(/^(\w+)\s*=\s*(.+)$/);
        if (!m) continue;
        const [, k, vRaw] = m;
        let v = vRaw.trim().replace(/,\s*\]/g, ']');
        if (k === 'name') cur.name = v.replace(/^"|"$/g, '');
        else { try { cur[k] = JSON.parse(v); } catch (e) { /* ignore */ } }
    }
    // Camera(name, matrix, dist, rvec, tvec, size) -- the app's own signature.
    return cams.map(c => new CameraCls(
        c.name, c.matrix, c.distortions, c.rotation, c.translation,
        c.size || [1920, 1200]));
}

function mapToObj(m) {
    if (!m) return null;
    if (m instanceof Map) {
        const o = {};
        for (const [k, v] of m.entries()) o[String(k)] = (v instanceof Map) ? mapToObj(v) : v;
        return o;
    }
    return m;
}

// A group is the app's per-identity bundle. Its exact shape has changed before,
// so probe rather than assume: emit {camName: instanceIdx} however it is stored.
function serializeGroup(g) {
    if (!g) return null;
    if (g instanceof Map) return mapToObj(g);
    if (Array.isArray(g)) return g.map(x => (x && x.camera !== undefined)
        ? { camera: x.camera, idx: x.instanceIdx !== undefined ? x.instanceIdx : x.idx }
        : x);
    const out = {};
    for (const k of Object.keys(g)) {
        const v = g[k];
        out[k] = (v instanceof Map) ? mapToObj(v) : v;
    }
    return out;
}

// --------------------------------------------------------------------------- //
function main() {
    const args = Object.fromEntries(process.argv.slice(2).reduce((a, v, i, arr) => {
        if (v.startsWith('--')) a.push([v.slice(2), arr[i + 1]]);
        return a;
    }, []));
    const data = JSON.parse(fs.readFileSync(args.input, 'utf8'));
    // Binary mode: one Float64Array for every keypoint in the session. Each
    // instance gets a SUBARRAY VIEW, which Instance() adopts by reference -- so a
    // 90,000-frame session costs one buffer, not 25M boxed [x,y] pairs.
    let blob = null;
    if (data.binary) {
        const buf = fs.readFileSync(data.binary);
        blob = new Float64Array(buf.buffer, buf.byteOffset, buf.byteLength / 8);
        const want = data.n_instances * data.n_nodes * 2;
        if (blob.length !== want) {
            throw new Error(`blob length ${blob.length} != n_instances*n_nodes*2 ${want}`);
        }
    }
    const numAnimals = parseInt(args['num-animals'] || data.n_animals, 10);

    const allCams = parseToml(fs.readFileSync(data.calibration, 'utf8'), Camera);
    const cameras = data.cameras.map(n => {
        const c = allCams.find(c => c.name === n);
        if (!c) throw new Error(`camera ${n} not in calibration`);
        return c;
    });

    const nodeNames = Array.from({ length: data.n_nodes }, (_, i) => 'n' + i);
    const skeleton = new sandbox.Skeleton('rat23', nodeNames, []);
    const tracks = Array.from({ length: numAnimals }, (_, i) => 'track_' + i);
    const session = new Session(cameras, skeleton, tracks, 'sdannce');
    sandbox.__session = session;

    // truth[frameOrdinal][camName] = [trueId per instance, in the SHUFFLED order]
    const truth = [];
    const K2 = data.n_nodes * 2;
    data.frames.forEach((fr, fi) => {
        const fg = new FrameGroup(fi);
        const t = {};
        const cams = blob ? fr.c : fr.cams;
        for (const cam of data.cameras) {
            const insts = cams[cam] || [];
            if (blob) {
                // [trueId, blobOffset] pairs
                t[cam] = insts.map(e => e[0]);
                insts.forEach((e, k) => {
                    const o = e[1] * K2;
                    fg.addInstance(cam, new Instance(blob.subarray(o, o + K2), k, 'predicted', 1.0));
                });
            } else {
                t[cam] = insts.map(i => i.id);
                insts.forEach((inst, k) => {
                    const pts = inst.pts.map(p => (p[0] == null ? null : [p[0], p[1]]));
                    fg.addInstance(cam, new Instance(pts, k, 'predicted', 1.0));
                });
            }
        }
        truth.push(t);
        session.addFrameGroup(fg);
    });

    // THE REAL TRACK ALL PATH. maxTargets is the animal count the user sets in
    // Track > number of animals.
    //
    // propagate=FALSE deliberately. propagateIdentitiesToTracks() rewrites every
    // instance's trackIdx to carry the identity and REBUILDS frameIdentityMap
    // against those new keys -- after which reading the map by the instance index
    // we constructed returns a trivial k -> k for every camera, which looks like a
    // perfect answer and is actually just the propagation's own bookkeeping. With
    // propagate=false the map keeps the keys commitTrackedFrame wrote, i.e. our
    // shuffled instance index, which is what the scoring needs. Association is
    // identical either way; propagation only rewrites labels for export/colour.
    // --fresh-per-frame: one INDEPENDENT tracker run per frame, i.e. what the
    // app's trackCurrentFrame does. Required for the human-label sets, whose
    // frames are sampled thousands apart: a temporal tracker asked to carry
    // identity across a 1,000-frame gap is not being tested, it is being
    // sabotaged, and its output says nothing about cross-view association. With
    // this flag the score is pure per-frame grouping (frame_acc); instance_acc
    // and switches are meaningless across independent runs and are not reported.
    const fresh = 'fresh-per-frame' in args;
    // --per-bout: one independent temporal run per close-interaction episode.
    // Bouts are minutes apart in the recording, so carrying tracker state between
    // them would be fiction; within a bout the temporal layer is exactly what is
    // under test.
    const perBout = 'per-bout' in args;
    // --params <json>: override the app's tracking thresholds through its OWN
    // setter, so the sweep exercises the same code path the Tracking Wizard does.
    // The shipped defaults are tuned for BMimica mice -- distanceThreshold 50 mm,
    // velocityThreshold 10 -- and TRIADS is rats in a larger arena, so the gates
    // are the first thing to suspect before concluding the tracker is weak.
    if (args.params) {
        const over = JSON.parse(fs.readFileSync(args.params, 'utf8'));
        if (typeof sandbox.setTrackingThresholds === 'function') {
            sandbox.setTrackingThresholds(over);
            const eff = sandbox.getTrackingThresholds();
            const shown = {};
            for (const k of Object.keys(over)) shown[k] = eff[k];
            console.error('[sdannce_track] requested ' + JSON.stringify(over));
            console.error('[sdannce_track] effective ' + JSON.stringify(shown));
        } else {
            throw new Error('setTrackingThresholds not exposed by the sandbox');
        }
    }
    const boutOf = data.frames.map(fr => (blob ? fr.b : fr.bout));
    const t0 = Date.now();
    let summary;
    if (fresh) {
        for (const fi of session.frameIndices) {
            summary = runCrossViewTracker(session, session.cameras, [fi], false, numAnimals);
        }
    } else if (perBout) {
        const byBout = new Map();
        session.frameIndices.forEach(fi => {
            const b = boutOf[fi];
            if (b == null) return;
            if (!byBout.has(b)) byBout.set(b, []);
            byBout.get(b).push(fi);
        });
        for (const [, list] of byBout) {
            summary = runCrossViewTracker(session, session.cameras, list, false, numAnimals);
        }
        console.error(`[sdannce_track] ${byBout.size} bouts`);
    } else {
        summary = runCrossViewTracker(session, session.cameras,
                                      session.frameIndices, false, numAnimals);
    }

    // Read identity back out of the session the way the app's own UI does:
    // frameIdentityMap keyed by (frameIdx, cameraName, rawTrackIdx). Instances
    // were constructed with trackIdx = their index in the SHUFFLED list, so this
    // recovers "which identity did the tracker give this detection".
    const out = [];
    data.frames.forEach((fr, fi) => {
        const assignments = {};
        for (const cam of data.cameras) {
            const insts = (blob ? fr.c : fr.cams)[cam] || [];
            for (let k = 0; k < insts.length; k++) {
                const id = session.getFrameIdentityValue(fi, cam, k);
                if (id !== undefined) assignments[`${cam}:${k}`] = id;
            }
        }
        out.push({ frame: fi, idx: blob ? fr.i : fr.idx, assignments, truth: truth[fi],
                   sep: blob ? fr.s : fr.sep, bout: blob ? fr.b : fr.bout });
    });
    const ms = Date.now() - t0;
    fs.writeFileSync(args.out, JSON.stringify({
        session: data.session, source: data.source, n_animals: numAnimals,
        cameras: data.cameras, ms, summary, frames: out,
        fresh_per_frame: fresh, per_bout: perBout,
        tag: data.tag || '', noise_px: data.noise_px || 0,
        dropout: data.dropout || 0, occlude_views: data.occlude_views || 0,
    }));
    console.error(`[sdannce_track] ${data.session}/${data.source}: ${out.length} frames in ${ms} ms`);
}

main();
