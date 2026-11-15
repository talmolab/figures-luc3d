#!/usr/bin/env node
// LUC3D headless Track All driver.
//
// Loads tracker.js + pose-data.js + triangulation.js into a Node vm sandbox,
// feeds it predictions from an aggregated camera H5 file for a single session,
// and writes per-frame identity assignments + triangulated 3D targets to JSON.

import { Command } from 'commander';
import * as h5 from 'h5wasm/node';
import vm from 'node:vm';
import fs from 'node:fs';
import path from 'node:path';

const LUCID_DIR = '/root/vast/eric/luc3d-bench/outputs/bartul_smoke/lucid_flat';

// ---------------------------------------------------------------------------
// Minimal TOML parser ported from lucid/file-io.js (parseCalibrationTOML +
// parseTOMLSection). Identical behaviour, no DOM.
// ---------------------------------------------------------------------------
function parseCalibrationTOML(text, CameraCtor) {
    const cameras = [];
    const sectionRegex = /^\[([^\]]+)\]\s*$/gm;
    const sections = [];
    let m;
    while ((m = sectionRegex.exec(text)) !== null) {
        sections.push({ name: m[1], start: m.index + m[0].length });
    }
    for (let i = 0; i < sections.length; i++) {
        const sec = sections[i];
        if (!sec.name.startsWith('cam_')) continue;
        const end = i + 1 < sections.length ? sections[i + 1].start : text.length;
        const body = text.substring(sec.start, end);
        const props = parseTOMLSection(body);
        const name = props.name || sec.name;
        const size = props.size || [640, 480];
        const matrix = props.matrix || [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        const dist = props.distortions || [0, 0, 0, 0, 0];
        const rvec = props.rotation || [0, 0, 0];
        const tvec = props.translation || [0, 0, 0];
        cameras.push(new CameraCtor(name, matrix, dist, rvec, tvec, size));
    }
    return cameras;
}
function parseTOMLSection(body) {
    const out = {};
    for (const raw of body.split('\n')) {
        const line = raw.trim();
        if (!line || line.startsWith('#') || line.startsWith('[')) continue;
        const eq = line.indexOf('=');
        if (eq === -1) continue;
        const key = line.substring(0, eq).trim();
        let value = line.substring(eq + 1).trim();
        if (value.startsWith('"') && value.endsWith('"')) { out[key] = value.slice(1, -1); continue; }
        value = value.replace(/,\s*\]/g, ']');
        try { out[key] = JSON.parse(value); } catch (e) { out[key] = value; }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Build the VM sandbox that mimics the browser globals tracker.js + pose-data.js
// + triangulation.js need.
// ---------------------------------------------------------------------------
// Replace hard-coded tracker.js constants with overrideable references to
// globalThis.TRACKER_PARAMS. Defaults preserved when the param is absent.
const TRACKER_PATCHES = [
    // crossViewScore weights (line ~108)
    {
        find: 'return 0.4 * epipolarScore(instA, camA, instB, camB) +\n           0.6 * reprojectionScore(instA, camA, instB, camB);',
        replace: 'return (globalThis.TRACKER_PARAMS?.wEpipolar ?? 0.4) * epipolarScore(instA, camA, instB, camB) +\n           (globalThis.TRACKER_PARAMS?.wReprojection ?? 0.6) * reprojectionScore(instA, camA, instB, camB);',
    },
    // reprojectionScore OKS sigma (line ~87)
    {
        find: 'var sigma2x2 = 2 * 20.0 * 20.0;',
        replace: 'var __sig = (globalThis.TRACKER_PARAMS?.reprojSigma ?? 20.0); var sigma2x2 = 2 * __sig * __sig;',
    },
    // matchPairwise score threshold (line ~416)
    {
        find: 'matches = matches.filter(function(m) { return m.score > 0.05; });',
        replace: 'matches = matches.filter(function(m) { return m.score > (globalThis.TRACKER_PARAMS?.scoreThresh ?? 0.05); });',
    },
    // reorderGroupsByPrevTargets — track continuity weight (line ~330)
    {
        find: 'score += 2.0 * (matchingTracks / totalTracks);',
        replace: 'score += (globalThis.TRACKER_PARAMS?.continuityWeight ?? 2.0) * (matchingTracks / totalTracks);',
    },
    // Third-camera reprojection cutoff (line ~483)
    {
        find: 'if (ii2 >= 0 && ii2 < insts3.length && cost3[gi2][ii2] < 100) {',
        replace: 'if (ii2 >= 0 && ii2 < insts3.length && cost3[gi2][ii2] < (globalThis.TRACKER_PARAMS?.thirdCamMaxDist ?? 100)) {',
    },
];

let __echoedParamsOnce = false;
function __echoParamsIfNeeded(trackerParams) {
    if (__echoedParamsOnce) return;
    __echoedParamsOnce = true;
    process.stderr.write(
        "[luc3d_track_all] loaded TRACKER_PARAMS = " +
        JSON.stringify(trackerParams || null) + "\n"
    );
}

function patchTrackerSource(src) {
    let out = src;
    for (const p of TRACKER_PATCHES) {
        if (!out.includes(p.find)) {
            throw new Error(`tracker.js patch did not match: ${p.find.slice(0, 60)}…`);
        }
        out = out.replace(p.find, p.replace);
    }
    return out;
}

function buildSandbox(trackerParams) {
    __echoParamsIfNeeded(trackerParams);
    const sandbox = {
        console, setTimeout, clearTimeout, Promise, Map, Set, Array, Object,
        Math, JSON, Number, String, Boolean, Error, TypeError, RangeError,
        parseInt, parseFloat, isNaN, isFinite, NaN, Infinity,
        WeakMap, WeakSet, Symbol, Proxy,
        Uint8Array, Float32Array, Float64Array, Int32Array, ArrayBuffer, DataView,
        RegExp, Date,
    };
    sandbox.globalThis = sandbox;
    sandbox.TRACKER_PARAMS = trackerParams || null;
    vm.createContext(sandbox);
    for (const f of ['triangulation.js', 'pose-data.js', 'tracker.js', 'kalman.js']) {
        let code = fs.readFileSync(path.join(LUCID_DIR, f), 'utf8');
        if (f === 'tracker.js') code = patchTrackerSource(code);
        vm.runInContext(code, sandbox, { filename: f });
    }
    // Classes declared with `class Foo {}` are lexical and not exposed as
    // sandbox properties — explicitly attach the symbols we need.
    vm.runInContext(`
        globalThis.Camera = Camera;
        globalThis.Instance = Instance;
        globalThis.FrameGroup = FrameGroup;
        globalThis.Session = Session;
        globalThis.matchFrameInstances = matchFrameInstances;
        globalThis.hungarianAlgorithm = hungarianAlgorithm;
        globalThis.reprojectPoints = reprojectPoints;
        globalThis.computeInstanceDistance = computeInstanceDistance;
        globalThis.triangulatePoints = triangulatePoints;
        // KalmanIdentityTracker + computeCentroidFromPoints3d are attached
        // to globalThis by kalman.js itself.

        // Missing-instance verification: in any camera with fewer visible
        // instances than numAnimals, re-assign each instance to its closest
        // 3D-reprojection identity via Hungarian. Robust to single-camera
        // occlusions where the default pairwise matching can swap IDs.
        globalThis.verifyUnderCountedCameras = function(frameGroup, cameras, camMap,
                                                        targets3d, currentAssignments,
                                                        numAnimals, params) {
            params = params || {};
            // Tunable thresholds (lower = stricter = more conservative reassignments)
            var maxAccept = params.uvMaxAcceptCost != null ? params.uvMaxAcceptCost : 120;
            var minMargin = params.uvMinMargin != null ? params.uvMinMargin : 1.2;
            if (!targets3d || targets3d.length === 0) return currentAssignments;
            var id3d = {};
            for (var ti = 0; ti < targets3d.length; ti++) {
                var t = targets3d[ti];
                if (t.identityId == null || !t.points3d) continue;
                id3d[t.identityId] = t.points3d;
            }
            var allIds = Object.keys(id3d).map(function(s){return parseInt(s);});
            if (allIds.length === 0) return currentAssignments;

            var fixed = new Map(currentAssignments);
            var camsVerified = 0, instancesReassigned = 0;

            for (var ci = 0; ci < cameras.length; ci++) {
                var cam = cameras[ci];
                var insts = (frameGroup.getInstances && frameGroup.getInstances(cam.name)) || [];
                if (insts.length === 0 || insts.length >= numAnimals) continue;
                camsVerified++;

                var nI = insts.length, nID = allIds.length;
                var cost = [];
                for (var i = 0; i < nI; i++) {
                    cost[i] = [];
                    for (var j = 0; j < nID; j++) {
                        var pts3d = id3d[allIds[j]];
                        var reproj = reprojectPoints(pts3d, cam.projectionMatrix);
                        var d = computeInstanceDistance(reproj, insts[i].points);
                        cost[i][j] = isFinite(d) ? d : 1e6;
                    }
                }
                var n = Math.max(nI, nID);
                while (cost.length < n) {
                    var pad = []; for (var k = 0; k < n; k++) pad.push(1e6); cost.push(pad);
                }
                for (var r = 0; r < cost.length; r++) {
                    while (cost[r].length < n) cost[r].push(1e6);
                }
                var assignment = hungarianAlgorithm(cost);
                // Apply for real instances only, with hypothesis testing:
                //   accept new id IFF cost is low (<= maxAccept) AND it beats
                //   the cost of the CURRENT assignment by >= minMargin (relative)
                for (var i2 = 0; i2 < nI; i2++) {
                    var j2 = assignment[i2];
                    if (j2 < 0 || j2 >= nID) continue;
                    var newCost = cost[i2][j2];
                    if (newCost > maxAccept) continue;
                    var key = cam.name + ':' + insts[i2].trackIdx;
                    var prevId = fixed.get(key);
                    var newId = allIds[j2];
                    if (prevId === newId) continue;
                    // Compute cost of the current (prev) assignment
                    var prevJ = allIds.indexOf(prevId);
                    var prevCost = prevJ >= 0 ? cost[i2][prevJ] : 1e6;
                    if (prevCost <= newCost * minMargin) continue;  // current is fine
                    fixed.set(key, newId);
                    instancesReassigned++;
                }
            }
            if (instancesReassigned > 0) {
                fixed.__stats = { camsVerified: camsVerified, instancesReassigned: instancesReassigned };
            }
            return fixed;
        };

        // ----------------------------------------------------------------
        // Clean-triangulation undercount verification (recommended over the
        // naive verifyUnderCountedCameras when fully-covered cams exist).
        //
        // Algorithm:
        //   1. Categorize cams into fullCams (|insts| >= numAnimals) and
        //      underCams (0 < |insts| < numAnimals).
        //   2. For each identity, gather its assigned instances in fullCams
        //      (these assignments are trustworthy because Hungarian had a
        //      bijection there).
        //   3. Re-triangulate each identity using ONLY those fullCam obs.
        //      This gives a "clean" 3D anchor not contaminated by any wrong
        //      under-count assignment.
        //   4. Project each clean 3D into each underCam, run Hungarian over
        //      (visible insts × all identities) on reprojection cost, accept
        //      override only when newCost <= maxAccept and prevCost >= newCost * minMargin.
        // ----------------------------------------------------------------
        globalThis.verifyUnderCountedCamerasClean = function(frameGroup, cameras,
                                                             currentAssignments, numAnimals, params) {
            params = params || {};
            var maxAccept = params.uvMaxAcceptCost != null ? params.uvMaxAcceptCost : 120;
            var minMargin = params.uvMinMargin != null ? params.uvMinMargin : 1.2;

            // 1. Categorize cams + collect their instances once.
            var fullCams = [], underCams = [];
            var camInsts = {};
            for (var ci = 0; ci < cameras.length; ci++) {
                var cam = cameras[ci];
                var insts = (frameGroup.getInstances && frameGroup.getInstances(cam.name)) || [];
                camInsts[cam.name] = insts;
                if (insts.length >= numAnimals) fullCams.push(cam);
                else if (insts.length > 0) underCams.push(cam);
            }
            if (fullCams.length < 2 || underCams.length === 0) return currentAssignments;

            // 2. Gather full-cam (cam, inst) observations per identity.
            var identityToObs = {};
            for (var fci = 0; fci < fullCams.length; fci++) {
                var fcam = fullCams[fci];
                var fInsts = camInsts[fcam.name];
                for (var ii = 0; ii < fInsts.length; ii++) {
                    var inst = fInsts[ii];
                    var key = fcam.name + ':' + inst.trackIdx;
                    var idnt = currentAssignments.get(key);
                    if (idnt == null) continue;
                    if (!identityToObs[idnt]) identityToObs[idnt] = [];
                    identityToObs[idnt].push({ cam: fcam, inst: inst });
                }
            }

            // 3. Triangulate each identity from its full-cam observations (need >= 2).
            var id3d = {};
            for (var idStr in identityToObs) {
                var obs = identityToObs[idStr];
                if (obs.length < 2) continue;
                var camsArr = obs.map(function(o){return o.cam;});
                var entries = obs.map(function(o){return o.inst;});
                var numKp = entries[0].points.length;
                var allObs = [];
                for (var k = 0; k < numKp; k++) {
                    var ptsObs = [];
                    for (var c = 0; c < entries.length; c++) {
                        var pt = entries[c].points[k];
                        ptsObs.push(pt && camsArr[c].undistortPoint ? camsArr[c].undistortPoint(pt) : pt);
                    }
                    allObs.push(ptsObs);
                }
                var pts3d = triangulatePoints(allObs, camsArr.map(function(c){return c.projectionMatrix;}));
                if (pts3d) id3d[parseInt(idStr)] = pts3d;
            }

            var allIds = Object.keys(id3d).map(function(s){return parseInt(s);});
            if (allIds.length === 0) return currentAssignments;

            // 4. For each undercount cam, Hungarian-verify against clean 3D.
            var fixed = new Map(currentAssignments);
            var camsVerified = 0, instancesReassigned = 0;
            for (var uci = 0; uci < underCams.length; uci++) {
                var ucam = underCams[uci];
                var insts3 = camInsts[ucam.name];
                camsVerified++;

                var nI = insts3.length, nID = allIds.length;
                var cost = [];
                for (var i3 = 0; i3 < nI; i3++) {
                    cost[i3] = [];
                    for (var j3 = 0; j3 < nID; j3++) {
                        var reproj = reprojectPoints(id3d[allIds[j3]], ucam.projectionMatrix);
                        var d = computeInstanceDistance(reproj, insts3[i3].points);
                        cost[i3][j3] = isFinite(d) ? d : 1e6;
                    }
                }
                var n = Math.max(nI, nID);
                while (cost.length < n) { var pad = []; for (var k2 = 0; k2 < n; k2++) pad.push(1e6); cost.push(pad); }
                for (var r = 0; r < cost.length; r++) while (cost[r].length < n) cost[r].push(1e6);
                var assignment = hungarianAlgorithm(cost);
                for (var i4 = 0; i4 < nI; i4++) {
                    var j4 = assignment[i4];
                    if (j4 < 0 || j4 >= nID) continue;
                    var newCost = cost[i4][j4];
                    if (newCost > maxAccept) continue;
                    var key2 = ucam.name + ':' + insts3[i4].trackIdx;
                    var prevId = fixed.get(key2);
                    var newId = allIds[j4];
                    if (prevId === newId) continue;
                    var prevJ = allIds.indexOf(prevId);
                    var prevCost = prevJ >= 0 ? cost[i4][prevJ] : 1e6;
                    if (prevCost <= newCost * minMargin) continue;
                    fixed.set(key2, newId);
                    instancesReassigned++;
                }
            }
            if (instancesReassigned > 0) {
                fixed.__stats = { camsVerified: camsVerified, instancesReassigned: instancesReassigned };
            }
            return fixed;
        };

        // ----------------------------------------------------------------
        // SORT-style 3D motion tracker for under-count verification.
        //
        // For each identity, maintain a constant-velocity state on the 3D
        // centroid PLUS a stored full 3D pose. On full-coverage frames we
        // re-triangulate cleanly (using only fully-covered cams) and update
        // the state. On under-counted frames we PREDICT (centroid + vel*dt)
        // and translate the stored pose to the predicted centroid — this
        // gives us the expected 3D pose even if the animal has been occluded
        // for several frames. Then we project that prediction into each
        // under-counted camera and run Hungarian (with margin gate).
        //
        // Why this is better than verifyUnderCountedCamerasClean:
        //   - Clean variant fails when ALL fully-covered cams also drop one
        //     animal in the same frame (no observation -> no triangulation).
        //   - Motion tracker keeps a confident estimate across short
        //     occlusions (up to motionMaxGap frames) using velocity.
        //   - Decouples "trust full-coverage observations" from "use the
        //     latest reliable estimate to disambiguate under-count cams".
        // ----------------------------------------------------------------
        globalThis.makeMotionTracker = function(params) {
            params = params || {};
            return {
                states: new Map(),  // identityId -> { pose, centroid, vel, lastFrame }
                alpha: params.motionAlpha != null ? params.motionAlpha : 0.5,
                maxGap: params.motionMaxGap != null ? params.motionMaxGap : 15,
            };
        };

        function _getCentroid(pts3d) {
            if (!pts3d) return null;
            var sx=0, sy=0, sz=0, n=0;
            for (var k = 0; k < pts3d.length; k++) {
                var p = pts3d[k];
                if (!p || !isFinite(p[0]) || !isFinite(p[1]) || !isFinite(p[2])) continue;
                sx += p[0]; sy += p[1]; sz += p[2]; n++;
            }
            return n === 0 ? null : [sx/n, sy/n, sz/n];
        }
        globalThis._getCentroid = _getCentroid;

        globalThis.updateMotionState = function(mt, identityId, pose3d, currentFrame) {
            var newCentroid = _getCentroid(pose3d);
            if (!newCentroid) return;
            var prev = mt.states.get(identityId);
            if (!prev) {
                mt.states.set(identityId, {
                    pose: pose3d.slice(),
                    centroid: newCentroid,
                    vel: [0,0,0],
                    lastFrame: currentFrame,
                });
                return;
            }
            var dt = Math.max(1, currentFrame - prev.lastFrame);
            var rawVel = [
                (newCentroid[0] - prev.centroid[0]) / dt,
                (newCentroid[1] - prev.centroid[1]) / dt,
                (newCentroid[2] - prev.centroid[2]) / dt,
            ];
            var a = mt.alpha;
            // Per-keypoint pose: prefer new (if non-null), else keep prev.
            var nK = Math.max(pose3d.length, prev.pose.length);
            var mergedPose = new Array(nK);
            for (var k = 0; k < nK; k++) {
                mergedPose[k] = pose3d[k] || prev.pose[k] || null;
            }
            mt.states.set(identityId, {
                pose: mergedPose,
                centroid: newCentroid,
                vel: [(1-a)*prev.vel[0] + a*rawVel[0],
                      (1-a)*prev.vel[1] + a*rawVel[1],
                      (1-a)*prev.vel[2] + a*rawVel[2]],
                lastFrame: currentFrame,
            });
        };

        // Predicted pose at currentFrame = stored pose translated by vel * dt.
        // Returns null when state is stale (dt > maxGap).
        globalThis.predictPoseAt = function(mt, identityId, currentFrame) {
            var s = mt.states.get(identityId);
            if (!s) return null;
            var dt = currentFrame - s.lastFrame;
            if (dt <= 0 || dt > mt.maxGap) return null;
            var dx = s.vel[0] * dt, dy = s.vel[1] * dt, dz = s.vel[2] * dt;
            var out = new Array(s.pose.length);
            for (var k = 0; k < s.pose.length; k++) {
                var p = s.pose[k];
                out[k] = p ? [p[0]+dx, p[1]+dy, p[2]+dz] : null;
            }
            return out;
        };

        // Re-triangulate each identity from FULLY-COVERED cameras. Returns
        // { id3d_clean: {id: pose3d}, fullCams: [...], underCams: [...] }.
        function _cleanTriangulateFromFullCams(frameGroup, cameras, currentAssignments, numAnimals) {
            var fullCams = [], underCams = [];
            var camInsts = {};
            for (var ci = 0; ci < cameras.length; ci++) {
                var cam = cameras[ci];
                var insts = (frameGroup.getInstances && frameGroup.getInstances(cam.name)) || [];
                camInsts[cam.name] = insts;
                if (insts.length >= numAnimals) fullCams.push(cam);
                else if (insts.length > 0) underCams.push(cam);
            }
            var id3d_clean = {};
            if (fullCams.length >= 2) {
                var identityToObs = {};
                for (var fci = 0; fci < fullCams.length; fci++) {
                    var fcam = fullCams[fci];
                    var fInsts = camInsts[fcam.name];
                    for (var ii = 0; ii < fInsts.length; ii++) {
                        var inst = fInsts[ii];
                        var key = fcam.name + ':' + inst.trackIdx;
                        var idnt = currentAssignments.get(key);
                        if (idnt == null) continue;
                        if (!identityToObs[idnt]) identityToObs[idnt] = [];
                        identityToObs[idnt].push({ cam: fcam, inst: inst });
                    }
                }
                for (var idStr in identityToObs) {
                    var obs = identityToObs[idStr];
                    if (obs.length < 2) continue;
                    var camsArr = obs.map(function(o){return o.cam;});
                    var entries = obs.map(function(o){return o.inst;});
                    var nK = entries[0].points.length;
                    var allObs = [];
                    for (var kk = 0; kk < nK; kk++) {
                        var ptsObs = [];
                        for (var c = 0; c < entries.length; c++) {
                            var pt = entries[c].points[kk];
                            ptsObs.push(pt && camsArr[c].undistortPoint ? camsArr[c].undistortPoint(pt) : pt);
                        }
                        allObs.push(ptsObs);
                    }
                    var pts3d = triangulatePoints(allObs, camsArr.map(function(c){return c.projectionMatrix;}));
                    if (pts3d) id3d_clean[parseInt(idStr)] = pts3d;
                }
            }
            return { id3d_clean: id3d_clean, fullCams: fullCams, underCams: underCams, camInsts: camInsts };
        }

        globalThis.verifyUnderCountedCamerasMotion = function(frameGroup, cameras,
                                                              currentAssignments, numAnimals,
                                                              motionTracker, currentFrame, params) {
            params = params || {};
            var maxAccept = params.uvMaxAcceptCost != null ? params.uvMaxAcceptCost : 120;
            var minMargin = params.uvMinMargin != null ? params.uvMinMargin : 1.2;
            var minAbsGap = params.uvMinAbsGap != null ? params.uvMinAbsGap : 0;
            var minAnimals = params.uvMinAnimals != null ? params.uvMinAnimals : 2;
            // Skip verification when too few animals — noise dominates on
            // easy sessions where baseline is already near-optimal.
            if (numAnimals < minAnimals) return currentAssignments;

            var t = _cleanTriangulateFromFullCams(frameGroup, cameras, currentAssignments, numAnimals);

            // Update motion state for identities we have a CLEAN observation of.
            for (var idStr in t.id3d_clean) {
                updateMotionState(motionTracker, parseInt(idStr), t.id3d_clean[idStr], currentFrame);
            }

            if (t.underCams.length === 0) return currentAssignments;

            // Build reference pose per identity. STRICT mode: only fire motion
            // verification when (a) the identity has NO clean obs from full
            // cams this frame, AND (b) motion state is fresh. Then we have a
            // genuinely occluded identity that motion prediction can help
            // disambiguate without contaminating frames where the current
            // observation is already the best reference. We never override
            // assignments for identities the full cams CAN see — those are
            // already trustworthy.
            //
            // Behavior: if clean cleared the identity (we already see it),
            // skip motion entirely for it. The under-count cam will still be
            // verified against clean obs only.
            var id3d_pred = {};
            for (var idStr3 in t.id3d_clean) id3d_pred[idStr3] = t.id3d_clean[idStr3];
            for (var idStr2 of motionTracker.states.keys()) {
                if (id3d_pred[idStr2] != null) continue;
                var predPose = predictPoseAt(motionTracker, idStr2, currentFrame);
                if (predPose) id3d_pred[idStr2] = predPose;
            }
            // If no identity needs motion (all have clean obs), this is a
            // no-op vs the clean variant. The motion path only contributes
            // when an identity is truly occluded across all full cams.
            var allIds = Object.keys(id3d_pred).map(function(s){return parseInt(s);});
            if (allIds.length === 0) return currentAssignments;

            // For each under-counted cam, Hungarian-verify against predicted pose reprojection.
            var fixed = new Map(currentAssignments);
            var camsVerified = 0, instancesReassigned = 0;
            for (var uci = 0; uci < t.underCams.length; uci++) {
                var ucam = t.underCams[uci];
                var insts3 = t.camInsts[ucam.name];
                camsVerified++;
                var nI = insts3.length, nID = allIds.length;
                var cost = [];
                for (var i3 = 0; i3 < nI; i3++) {
                    cost[i3] = [];
                    for (var j3 = 0; j3 < nID; j3++) {
                        var reproj = reprojectPoints(id3d_pred[allIds[j3]], ucam.projectionMatrix);
                        var d = computeInstanceDistance(reproj, insts3[i3].points);
                        cost[i3][j3] = isFinite(d) ? d : 1e6;
                    }
                }
                var n = Math.max(nI, nID);
                while (cost.length < n) { var pad = []; for (var k2 = 0; k2 < n; k2++) pad.push(1e6); cost.push(pad); }
                for (var r = 0; r < cost.length; r++) while (cost[r].length < n) cost[r].push(1e6);
                var assignment = hungarianAlgorithm(cost);
                for (var i4 = 0; i4 < nI; i4++) {
                    var j4 = assignment[i4];
                    if (j4 < 0 || j4 >= nID) continue;
                    var newCost = cost[i4][j4];
                    if (newCost > maxAccept) continue;
                    var key2 = ucam.name + ':' + insts3[i4].trackIdx;
                    var prevId = fixed.get(key2);
                    var newId = allIds[j4];
                    if (prevId === newId) continue;
                    var prevJ = allIds.indexOf(prevId);
                    var prevCost = prevJ >= 0 ? cost[i4][prevJ] : 1e6;
                    if (prevCost <= newCost * minMargin) continue;       // ratio gate
                    if (prevCost - newCost < minAbsGap) continue;        // absolute gate
                    fixed.set(key2, newId);
                    instancesReassigned++;
                }
            }
            if (instancesReassigned > 0) {
                fixed.__stats = { camsVerified: camsVerified, instancesReassigned: instancesReassigned };
            }
            return fixed;
        };

        // ----------------------------------------------------------------
        // Variance-inflation-style soft propagation block.
        //
        // Inspired by Lightning Pose 3D's mvEKS variance inflation
        // (Aharon, Whiteway et al., bioRxiv 2026, Sec. 4.10): when a
        // single view's prediction is geometrically inconsistent with
        // the consensus of the other views, the right move is to
        // *down-weight* its influence rather than overwrite it.
        //
        // Here the analogue is: for each (cam:trackIdx -> identity) entry
        // in an under-counted camera, triangulate the assigned identity
        // from ONLY the fully-covered cameras and reproject into this
        // camera. If the reprojection is far from the instance AND a
        // different identity reprojects much closer, this view's vote
        // for the next frame's prior is suspect.
        //
        // ACTION: drop that key from the assignments map that gets
        // forwarded as prevAssignments. The tracker's OUTPUT (what
        // motmetrics scores) keeps the assignment unchanged. Only the
        // next-frame voting prior loses a suspect vote — and the other
        // cams' (consistent) votes still drive the group's identity.
        //
        // This is monotonic relative to baseline: in the worst case we
        // remove a vote that happened to be correct, and the other
        // cameras' votes carry the decision. We never introduce a wrong
        // assignment, which is the failure mode of the override variants.
        // ----------------------------------------------------------------
        globalThis.filterInconsistentAssignments = function(frameGroup, cameras,
                                                            assignments, numAnimals, params) {
            params = params || {};
            var viAbsThresh = params.viAbsThresh != null ? params.viAbsThresh : 60;
            var viRelThresh = params.viRelThresh != null ? params.viRelThresh : 0.6;

            // Categorize cams
            var fullCams = [], underCams = [];
            var camInsts = {};
            for (var ci = 0; ci < cameras.length; ci++) {
                var cam = cameras[ci];
                var insts = (frameGroup.getInstances && frameGroup.getInstances(cam.name)) || [];
                camInsts[cam.name] = insts;
                if (insts.length >= numAnimals) fullCams.push(cam);
                else if (insts.length > 0) underCams.push(cam);
            }
            if (fullCams.length < 2 || underCams.length === 0) return assignments;

            // Clean triangulation per identity from full cams
            var identityToObs = {};
            for (var fci = 0; fci < fullCams.length; fci++) {
                var fcam = fullCams[fci];
                var fInsts = camInsts[fcam.name];
                for (var ii = 0; ii < fInsts.length; ii++) {
                    var inst = fInsts[ii];
                    var key0 = fcam.name + ':' + inst.trackIdx;
                    var idnt = assignments.get(key0);
                    if (idnt == null) continue;
                    if (!identityToObs[idnt]) identityToObs[idnt] = [];
                    identityToObs[idnt].push({ cam: fcam, inst: inst });
                }
            }
            var id3d = {};
            for (var idStr in identityToObs) {
                var obs = identityToObs[idStr];
                if (obs.length < 2) continue;
                var camsArr = obs.map(function(o){return o.cam;});
                var entries = obs.map(function(o){return o.inst;});
                var nK = entries[0].points.length;
                var allObs = [];
                for (var k = 0; k < nK; k++) {
                    var ptsObs = [];
                    for (var c = 0; c < entries.length; c++) {
                        var pt = entries[c].points[k];
                        ptsObs.push(pt && camsArr[c].undistortPoint ? camsArr[c].undistortPoint(pt) : pt);
                    }
                    allObs.push(ptsObs);
                }
                var pts3d = triangulatePoints(allObs, camsArr.map(function(c){return c.projectionMatrix;}));
                if (pts3d) id3d[parseInt(idStr)] = pts3d;
            }
            if (Object.keys(id3d).length < 2) return assignments;

            // For each under-counted cam, check each visible inst's assignment
            var filtered = new Map(assignments);
            var checked = 0, dropped = 0;
            for (var uci = 0; uci < underCams.length; uci++) {
                var ucam = underCams[uci];
                var insts3 = camInsts[ucam.name];
                for (var i = 0; i < insts3.length; i++) {
                    var key = ucam.name + ':' + insts3[i].trackIdx;
                    var assignedId = assignments.get(key);
                    if (assignedId == null) continue;
                    var pts3d_assigned = id3d[assignedId];
                    if (!pts3d_assigned) continue;
                    checked++;
                    var reproj = reprojectPoints(pts3d_assigned, ucam.projectionMatrix);
                    var costAssigned = computeInstanceDistance(reproj, insts3[i].points);
                    if (!isFinite(costAssigned) || costAssigned <= viAbsThresh) continue;
                    // Find best alternative
                    var costBest = Infinity;
                    for (var idStr2 in id3d) {
                        var idInt = parseInt(idStr2);
                        if (idInt === assignedId) continue;
                        var reproj2 = reprojectPoints(id3d[idInt], ucam.projectionMatrix);
                        var c2 = computeInstanceDistance(reproj2, insts3[i].points);
                        if (isFinite(c2) && c2 < costBest) costBest = c2;
                    }
                    // Drop only if alternative is much better than assigned
                    if (costBest < costAssigned * viRelThresh) {
                        filtered.delete(key);
                        dropped++;
                    }
                }
            }
            if (dropped > 0) {
                filtered.__stats = { checked: checked, dropped: dropped };
            }
            return filtered;
        };
    `, sandbox, { filename: '__exports.js' });
    return sandbox;
}

// ---------------------------------------------------------------------------
// Load aggregated camera H5: returns Float64Array of shape (n_frames, n_animals, n_nodes, 2)
// for the requested session.
// ---------------------------------------------------------------------------
function sliceCamSession(h5File, sessionIdx) {
    const ds = h5File.get('tracks');
    const [, nF, nA, nN, two] = ds.shape;
    // Hyperslab read — only this session, not the whole dataset (h5wasm WASM heap is 2 GB).
    const data = ds.slice([[sessionIdx, sessionIdx + 1], [0, nF], [0, nA], [0, nN], [0, two]]);
    return { data, nFrames: nF, nAnimals: nA, nNodes: nN };
}

function instancePoints(sub, fi, animal, nFrames, nAnimals, nNodes) {
    const pts = new Array(nNodes);
    let allNull = true;
    for (let k = 0; k < nNodes; k++) {
        const base = (((fi * nAnimals) + animal) * nNodes + k) * 2;
        const x = sub[base];
        const y = sub[base + 1];
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
            pts[k] = null;
        } else {
            pts[k] = [x, y];
            allNull = false;
        }
    }
    return allNull ? null : pts;
}

// ---------------------------------------------------------------------------
// Constant-velocity 3D Kalman predictor (opt-in via TRACKER_PARAMS.applyKalman).
// State per identity: last two observed 3D point arrays. Predict by extrapolating
// each keypoint using the per-keypoint velocity (currPts - prevPts) / dt, then
// stepping forward by (currentFrame - lastFrame). When fewer than 2 obs exist,
// returns input unchanged (no motion model).
// ---------------------------------------------------------------------------
function makeKalmanState() { return new Map(); }

function kalmanPredictTargets(prevTargets3d, kalmanState, currentFrame, alpha = 1.0) {
    if (!prevTargets3d || prevTargets3d.length === 0) return prevTargets3d;
    return prevTargets3d.map(t => {
        const id = t.identityId;
        if (id == null) return t;
        const s = kalmanState.get(id);
        if (!s || !s.prevPts || !s.currPts || !t.points3d) return t;
        const dt = Math.max(1, currentFrame - s.lastFrame);
        const stepN = Math.min(dt, 10);  // cap to avoid runaway extrapolation across long gaps
        const predicted = t.points3d.map((pt, k) => {
            if (!pt || !s.prevPts[k] || !s.currPts[k]) return pt;
            const dtPrev = Math.max(1, s.lastFrame - s.prevFrame);
            const vx = (s.currPts[k][0] - s.prevPts[k][0]) / dtPrev;
            const vy = (s.currPts[k][1] - s.prevPts[k][1]) / dtPrev;
            const vz = (s.currPts[k][2] - s.prevPts[k][2]) / dtPrev;
            return [pt[0] + alpha * vx * stepN,
                    pt[1] + alpha * vy * stepN,
                    pt[2] + alpha * vz * stepN];
        });
        return { ...t, points3d: predicted };
    });
}

function kalmanUpdate(targets3d, kalmanState, frame) {
    if (!targets3d) return;
    for (const t of targets3d) {
        if (t.identityId == null || !t.points3d) continue;
        const prev = kalmanState.get(t.identityId);
        kalmanState.set(t.identityId, {
            prevPts: prev ? prev.currPts : null,
            prevFrame: prev ? prev.lastFrame : frame,
            currPts: t.points3d,
            lastFrame: frame,
        });
    }
}

// ---------------------------------------------------------------------------
// Offline tracklet stitching (opt-in via TRACKER_PARAMS.applyStitching).
// Two heuristics that run after the full forward pass over perFrame:
//   (a) Single-frame identity flips on the same (cam, raw_track_idx) key
//       are reverted (X -> Y -> X within stitchMaxGap frames).
//   (b) Multi-frame swap consensus: if (cam, raw_track_idx) key consistently
//       maps to id A but in a window had id B with shorter run length, prefer A.
// Defaults are conservative — only undoes flips clearly inconsistent with their
// own surrounding context.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Interpolation across short gaps (opt-in via TRACKER_PARAMS.applyInterpolation).
// For each (cam, raw_track_idx) key that has assignments at frames A and B with
// A < B and B - A <= interpMaxGap, fill in the assignments for intermediate
// frames as the same identity. Only fills if the same identity at A == identity at B.
// This won't change IDSW (no new switches introduced) but should improve IDR by
// recovering ID predictions where the tracker dropped them.
// ---------------------------------------------------------------------------
function interpolateAssignments(perFrame, maxGap = 5) {
    if (!perFrame || perFrame.length === 0) return perFrame;
    // Build per-key sparse timeline of (frame, idx_in_perFrame, identityId)
    const timelines = new Map();
    for (let fi = 0; fi < perFrame.length; fi++) {
        const fr = perFrame[fi];
        for (const [key, id] of fr.assignments) {
            if (!timelines.has(key)) timelines.set(key, []);
            timelines.get(key).push({ pfIdx: fi, frame: fr.frame, id });
        }
    }
    let filled = 0;
    for (const [key, entries] of timelines) {
        for (let i = 0; i < entries.length - 1; i++) {
            const a = entries[i], b = entries[i + 1];
            if (a.id !== b.id) continue;  // would introduce a switch
            const frameGap = b.frame - a.frame;
            if (frameGap <= 1) continue;  // already contiguous
            if (frameGap > maxGap) continue;  // too far to interpolate
            // Fill all frames strictly between pfIdx a and b for this key.
            for (let mid = a.pfIdx + 1; mid < b.pfIdx; mid++) {
                const fr = perFrame[mid];
                // Only fill if this key has no entry at this frame
                let already = false;
                for (const [k2, _] of fr.assignments) if (k2 === key) { already = true; break; }
                if (already) continue;
                fr.assignments.push([key, a.id]);
                filled++;
            }
        }
    }
    if (filled > 0) console.error(`[luc3d][interp] filled ${filled} assignment slots across short gaps`);
    return perFrame;
}

function stitchPerFrame(perFrame, maxGap = 5) {
    if (!perFrame || perFrame.length === 0) return perFrame;
    // Build per-key timeline (preserves event order; frames are in ascending order in perFrame).
    const timelines = new Map();
    for (const fr of perFrame) {
        for (const [key, id] of fr.assignments) {
            if (!timelines.has(key)) timelines.set(key, []);
            timelines.get(key).push({ frame: fr.frame, id });
        }
    }

    // Compute overrides
    const overrides = new Map();   // `${frame}:${key}` -> newId
    for (const [key, entries] of timelines) {
        for (let i = 1; i < entries.length - 1; i++) {
            const prev = entries[i - 1], curr = entries[i], next = entries[i + 1];
            // Pattern: X -> Y -> X (Y different, span <= maxGap frames). Revert.
            if (prev.id !== curr.id && next.id !== curr.id && prev.id === next.id &&
                next.frame - prev.frame <= maxGap) {
                overrides.set(`${curr.frame}:${key}`, prev.id);
            }
        }
    }
    // 2nd pass: also undo flips of length 2 (X -> Y -> Y -> X) within maxGap.
    for (const [key, entries] of timelines) {
        for (let i = 1; i < entries.length - 2; i++) {
            const a = entries[i - 1], b = entries[i], c = entries[i + 1], d = entries[i + 2];
            if (a.id === d.id && b.id === c.id && a.id !== b.id &&
                d.frame - a.frame <= maxGap) {
                overrides.set(`${b.frame}:${key}`, a.id);
                overrides.set(`${c.frame}:${key}`, a.id);
            }
        }
    }

    if (overrides.size === 0) return perFrame;

    let applied = 0;
    for (const fr of perFrame) {
        fr.assignments = fr.assignments.map(([key, id]) => {
            const ov = overrides.get(`${fr.frame}:${key}`);
            if (ov != null && ov !== id) { applied++; return [key, ov]; }
            return [key, id];
        });
    }
    console.error(`[luc3d][stitch] reverted ${applied} per-key identity flips`);
    return perFrame;
}

// ---------------------------------------------------------------------------
// Temporal 3D PAF helpers (Landings 2 + 3 — hypothesis enumeration + one-
// frame lookahead). Landing 1 (KF Mahalanobis added to per-frame Hungarian
// as Signal 5) lives inside tracker.js::reorderGroupsByPrevTargets.
//
// Landing 2: after the standard match, if the KF's predicted distribution
// strongly disagrees with the matched assignment (Mahalanobis² > gate), try
// every permutation of (identity → group) on the *current* frame's groups
// and pick the one with the lowest total Mahalanobis². This is the
// hypothesis-testing piece — the "soft" PAF idea applied to identity
// assignment.
//
// Landing 3: maintain a one-frame buffer of `(frameIdx, perFrame record,
// preMatchKF state)`. When processing frame t, retrospectively re-score the
// buffered frame t-1's assignment using frame t's evidence: if a swap at
// t-1 would have produced a more consistent t-1→t flow integral, retroactively
// flip the buffered assignment. This is the forward-flow leg of the
// PAF analog (the integral over t-1 → t → t+1).
// ---------------------------------------------------------------------------

function _allPermutations(n) {
    // Returns all permutations of [0..n-1] as arrays.
    if (n <= 1) return [[0]];
    const out = [];
    const recur = (arr, rest) => {
        if (rest.length === 0) { out.push(arr); return; }
        for (let i = 0; i < rest.length; i++) {
            const next = rest.slice(0, i).concat(rest.slice(i + 1));
            recur(arr.concat([rest[i]]), next);
        }
    };
    recur([], Array.from({ length: n }, (_, i) => i));
    return out;
}

function _mahaMatrix(sandbox, kf, identityIds, groupCentroids, frameIdx) {
    // matrix[i][g] = Mahalanobis²(identityIds[i] predicted at frameIdx,
    //                              groupCentroids[g] observation)
    const M = [];
    for (let i = 0; i < identityIds.length; i++) {
        M[i] = [];
        for (let g = 0; g < groupCentroids.length; g++) {
            if (!groupCentroids[g]) { M[i][g] = Infinity; continue; }
            const m = kf.mahalanobisSq(identityIds[i], groupCentroids[g], frameIdx);
            M[i][g] = (m != null && isFinite(m)) ? m : Infinity;
        }
    }
    return M;
}

function _permTotalCost(perm, costMat) {
    // perm[i] = group index assigned to identity i. Returns sum of costs;
    // Infinity short-circuits.
    let total = 0;
    for (let i = 0; i < perm.length; i++) {
        const c = costMat[i][perm[i]];
        if (!isFinite(c)) return Infinity;
        total += c;
    }
    return total;
}

/**
 * Landing 2 — hypothesis-swap rescue.
 *
 * If KF Mahalanobis² of any (current_identity, current_group) exceeds the
 * gate, try every permutation of identity-to-group assignment among the
 * existing groups and pick the lowest-total-Mahalanobis one. Emits the new
 * assignment as a Map plus updated identityIds on targets3d.
 *
 * Returns { assignments, targets3d, swapped } where:
 *   - assignments: Map<"cam:trackIdx", identityId> (possibly with swaps applied)
 *   - targets3d: same shape as r.targets3d but with possibly-updated identityId
 *   - swapped: bool, did we actually change anything
 */
function rescueByHypothesis(sandbox, r, kf, frameIdx, numAnimals, params) {
    params = params || {};
    // Stricter knobs: only commit a swap if the BEST alternative permutation
    // has total cost <= currentCost * swapMarginInv. swapMarginInv default 0.7
    // = need >=30% improvement to flip.
    const swapMarginInv = params.pafSwapMarginInv != null ? params.pafSwapMarginInv : 0.7;
    if (!kf || !numAnimals || numAnimals < 2) return { ...r, swapped: false };
    const groups = r.targets3d || [];
    if (groups.length < 2) return { ...r, swapped: false };

    // Restrict to groups WITH an assigned identityId AND a usable centroid.
    const activeIdxs = [];
    const centroids = [];
    const idsCurr = [];
    for (let g = 0; g < groups.length; g++) {
        const t = groups[g];
        if (t.identityId == null) continue;
        const c = t.points3d ? sandbox.computeCentroidFromPoints3d(t.points3d) : null;
        if (!c) continue;
        if (!kf.hasIdentity(t.identityId)) continue;  // KF has no prediction yet
        activeIdxs.push(g);
        centroids.push(c);
        idsCurr.push(t.identityId);
    }
    if (activeIdxs.length < 2) return { ...r, swapped: false };

    // M[i][g] = Mahalanobis² of identity idsCurr[i] under groupCentroid g.
    const M = _mahaMatrix(sandbox, kf, idsCurr, centroids, frameIdx);

    // Current permutation: identity i was assigned to group i (since we
    // built idsCurr in the same order as activeIdxs/centroids).
    const identityPerm = idsCurr.map((_, i) => i);
    const currentCost = _permTotalCost(identityPerm, M);

    // Gate: only rescue if at least one diagonal cost exceeds the KF gate.
    let maxDiag = 0;
    for (let i = 0; i < idsCurr.length; i++) if (M[i][i] > maxDiag) maxDiag = M[i][i];
    if (maxDiag < kf.gateSq) return { ...r, swapped: false };

    // Enumerate permutations and pick the lowest cost.
    let bestPerm = identityPerm, bestCost = currentCost;
    const perms = _allPermutations(idsCurr.length);
    for (const perm of perms) {
        const c = _permTotalCost(perm, M);
        if (c < bestCost) { bestCost = c; bestPerm = perm; }
    }
    if (bestPerm === identityPerm || _permEqual(bestPerm, identityPerm)) {
        return { ...r, swapped: false };
    }
    // Require best to be SIGNIFICANTLY better than current (avoid noise-driven flips).
    if (bestCost > currentCost * swapMarginInv) {
        return { ...r, swapped: false };
    }

    // Apply the swap. For each identity i (in idsCurr order), it now goes to
    // groups[activeIdxs[bestPerm[i]]] instead of groups[activeIdxs[i]].
    const newTargets3d = groups.map(t => ({ ...t }));
    const newAssignments = new Map();
    // Carry forward any assignments for inactive groups (no swap there).
    for (const [k, v] of r.assignments) newAssignments.set(k, v);

    for (let i = 0; i < idsCurr.length; i++) {
        const oldGroupIdx = activeIdxs[i];
        const newGroupIdx = activeIdxs[bestPerm[i]];
        if (oldGroupIdx === newGroupIdx) continue;
        newTargets3d[newGroupIdx].identityId = idsCurr[i];
        // Update per-instance assignments for the new group's instances.
        const newPrevInsts = newTargets3d[newGroupIdx].prevInstances;
        if (newPrevInsts && newPrevInsts.forEach) {
            newPrevInsts.forEach((inst, camName) => {
                newAssignments.set(camName + ':' + inst.trackIdx, idsCurr[i]);
            });
        }
    }
    return { ...r, targets3d: newTargets3d, assignments: newAssignments, swapped: true };
}

function _permEqual(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
}

/**
 * Landing 3 — one-frame lookahead retroactive flip.
 *
 * Given the *current* frame's matched groups (frameT) and the *buffered*
 * previous frame's record (frameTm1), score whether the previous frame's
 * identity assignments are consistent with the current frame's centroids.
 * If the KF state at end-of-frame-t-1 (BEFORE update) predicts frame-t's
 * centroids better under a swapped assignment at t-1, retroactively flip
 * frame t-1.
 *
 * This implements the "forward flow integral" leg: at t-1 we couldn't
 * decide; at t we get new evidence and can finalize t-1.
 *
 * The retroactive flip ALSO requires re-updating the KF — but the KF is a
 * forward-only filter, so the cleanest implementation is to NEVER call
 * kf.update() until the frame is finalized. The driver does this by
 * deferring kf.update() by one frame: at frame t we finalize frame t-1's
 * assignment using frame t's evidence, then update KF with frame t-1's
 * (now-finalized) centroids.
 *
 * For minimal complexity, this landing only flips the most recent buffered
 * frame; it does not propagate flips backwards further than one frame.
 */
function lookaheadFlip(sandbox, bufferedRecord, currentFrameTargets3d, kf, currentFrameIdx, params) {
    params = params || {};
    // Require strong evidence to retroactively flip — much stricter than L2.
    const flipMinAbsDelta = params.pafFlipMinAbsDelta != null ? params.pafFlipMinAbsDelta : 5;
    const flipMaxRatio    = params.pafFlipMaxRatio    != null ? params.pafFlipMaxRatio    : 0.5;
    if (!bufferedRecord || !kf) return { record: bufferedRecord, flipped: false };
    const t3d_t = currentFrameTargets3d || [];
    if (t3d_t.length < 2) return { record: bufferedRecord, flipped: false };

    // Compute current-frame centroids per identity at frame t.
    const currCentroidsByIdent = new Map();
    for (const t of t3d_t) {
        if (t.identityId == null || !t.points3d) continue;
        const c = sandbox.computeCentroidFromPoints3d(t.points3d);
        if (c) currCentroidsByIdent.set(t.identityId, c);
    }
    if (currCentroidsByIdent.size < 2) return { record: bufferedRecord, flipped: false };

    // Frame t-1's "candidate assignments" — only the targets3d entries with
    // identityId present and valid centroid.
    const tm1_t3d = bufferedRecord.targets3d || [];
    const tm1Items = [];
    for (const t of tm1_t3d) {
        if (t.identityId == null || !t.points3d) continue;
        const c = sandbox.computeCentroidFromPoints3d(t.points3d);
        if (c) tm1Items.push({ id: t.identityId, centroid: c, targetEntry: t });
    }
    if (tm1Items.length < 2) return { record: bufferedRecord, flipped: false };

    // The flow integral over t-1 → t for the current assignment is the sum
    // over each identity i of |c_t-1[i] - c_t[i]|² / σ² (constant-velocity
    // model with σ = sqrt(KF rObs)). Compare against swapped permutations.
    const ids = tm1Items.map(it => it.id);
    const centroidsTm1 = tm1Items.map(it => it.centroid);
    const flowCost = (perm) => {
        // perm[i] = which t-1 centroid is assigned to identity ids[i] at t.
        let total = 0;
        for (let i = 0; i < perm.length; i++) {
            const cT = currCentroidsByIdent.get(ids[i]);
            const cTm1 = centroidsTm1[perm[i]];
            if (!cT || !cTm1) { total += 1e9; continue; }
            const dx = cT[0] - cTm1[0], dy = cT[1] - cTm1[1], dz = cT[2] - cTm1[2];
            total += (dx * dx + dy * dy + dz * dz) / kf.rObs;
        }
        return total;
    };

    const identityPerm = ids.map((_, i) => i);
    const currentCost = flowCost(identityPerm);
    let bestPerm = identityPerm, bestCost = currentCost;
    const perms = _allPermutations(ids.length);
    for (const perm of perms) {
        const c = flowCost(perm);
        if (c < bestCost) { bestCost = c; bestPerm = perm; }
    }
    if (_permEqual(bestPerm, identityPerm)) return { record: bufferedRecord, flipped: false };

    // Require a meaningful improvement to commit the flip — defaults are
    // 50% lower flow cost AND absolute Δ ≥ 5 (Mahalanobis units). Very
    // conservative; we'd rather miss a flip than retroactively introduce
    // a switch. Tunable via pafFlipMaxRatio / pafFlipMinAbsDelta.
    if (currentCost - bestCost < flipMinAbsDelta || bestCost > currentCost * flipMaxRatio) {
        return { record: bufferedRecord, flipped: false };
    }

    // Apply the swap to bufferedRecord. perm[i] = which t-1 item should now
    // get identity ids[i].
    const newTargets3d = (bufferedRecord.targets3d || []).map(t => ({ ...t }));
    const newAssignments = new Map(bufferedRecord.assignments);
    // We need to find the index of tm1Items[i].targetEntry inside newTargets3d.
    const entryIdxOf = (entry) => newTargets3d.indexOf(entry);

    for (let i = 0; i < ids.length; i++) {
        const targetIdx = entryIdxOf(tm1Items[bestPerm[i]].targetEntry);
        if (targetIdx < 0) continue;
        newTargets3d[targetIdx].identityId = ids[i];
        const prevInsts = newTargets3d[targetIdx].prevInstances;
        if (prevInsts && prevInsts.forEach) {
            prevInsts.forEach((inst, camName) => {
                newAssignments.set(camName + ':' + inst.trackIdx, ids[i]);
            });
        }
    }
    return {
        record: { ...bufferedRecord, targets3d: newTargets3d, assignments: newAssignments },
        flipped: true,
    };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
    const program = new Command();
    program
        .requiredOption('--session-idx <n>', 'session row index in master sheet', v => parseInt(v))
        .requiredOption('--num-animals <n>', 'expected number of animals', v => parseInt(v))
        .requiredOption('--calibration <path>', 'calibration.toml path')
        .requiredOption('--pred-h5-dir <path>', 'directory containing {cam}_predictions.h5 files')
        .requiredOption('--out <path>', 'output JSON path')
        .option('--cameras <list>', 'comma-separated camera names', 'back,backL,mid,midL,top,topL')
        .option('--max-frames <n>', 'cap frames processed (debugging)', v => parseInt(v))
        .option('--params <path>', 'JSON file with TRACKER_PARAMS overrides');
    program.parse();
    const opts = program.opts();
    const cameras = opts.cameras.split(',').map(s => s.trim());

    let trackerParams = null;
    if (opts.params) trackerParams = JSON.parse(fs.readFileSync(opts.params, 'utf8'));

    await h5.ready;
    const sandbox = buildSandbox(trackerParams);
    const { Camera, Session, FrameGroup, Instance, matchFrameInstances } = sandbox;

    // 1. Parse calibration -> ordered list of Camera objects, filter to bench cameras
    const tomlText = fs.readFileSync(opts.calibration, 'utf8');
    const allCams = parseCalibrationTOML(tomlText, Camera);
    const benchCams = cameras.map(name => {
        const c = allCams.find(c => c.name === name);
        if (!c) throw new Error(`Camera ${name} missing in calibration`);
        return c;
    });

    // 2. Open all cam H5s, slice the requested session
    const camSlices = {};
    let minFrames = Infinity, nAnimals = 0, nNodes = 0;
    for (const cam of cameras) {
        const h5path = path.join(opts.predH5Dir, `${cam}_predictions.h5`);
        const f = new h5.File(h5path, 'r');
        const sl = sliceCamSession(f, opts.sessionIdx);
        camSlices[cam] = sl;
        f.close();
        minFrames = Math.min(minFrames, sl.nFrames);
        nAnimals = Math.max(nAnimals, sl.nAnimals);
        nNodes = Math.max(nNodes, sl.nNodes);
    }
    let frameLimit = minFrames;
    if (opts.maxFrames) frameLimit = Math.min(frameLimit, opts.maxFrames);

    // 3. Determine which frames actually have data (any cam has any non-null instance)
    const usedFrames = [];
    for (let fi = 0; fi < frameLimit; fi++) {
        let any = false;
        for (const cam of cameras) {
            const sl = camSlices[cam];
            for (let a = 0; a < sl.nAnimals && !any; a++) {
                const p = instancePoints(sl.data, fi, a, sl.nFrames, sl.nAnimals, sl.nNodes);
                if (p) any = true;
            }
            if (any) break;
        }
        if (any) usedFrames.push(fi);
    }
    console.error(`[luc3d] session=${opts.sessionIdx} frames-with-data=${usedFrames.length}/${frameLimit} numAnimals=${opts.numAnimals}`);

    // 4. Build Session + FrameGroups
    const tracks = Array.from({ length: nAnimals }, (_, i) => 'track_' + i);
    const session = new Session(benchCams, { nodes: [] }, tracks, 'bench');
    for (const fi of usedFrames) {
        const fg = new FrameGroup(fi);
        for (const cam of cameras) {
            const sl = camSlices[cam];
            for (let a = 0; a < sl.nAnimals; a++) {
                const pts = instancePoints(sl.data, fi, a, sl.nFrames, sl.nAnimals, sl.nNodes);
                if (!pts) continue;
                fg.addInstance(cam, new Instance(pts, a, 'predicted', 1.0));
            }
        }
        session.addFrameGroup(fg);
    }

    // 5. Drive matchFrameInstances per frame (mirrors trackAll() in index.html:16532)
    const applyKalman = !!(trackerParams && trackerParams.applyKalman);
    const kalmanAlpha = (trackerParams && trackerParams.kalmanAlpha != null) ? trackerParams.kalmanAlpha : 1.0;
    const applyStitching = !!(trackerParams && trackerParams.applyStitching);
    const stitchMaxGap = (trackerParams && trackerParams.stitchMaxGap != null) ? trackerParams.stitchMaxGap : 5;
    const applyUndercountVerify = !!(trackerParams && trackerParams.applyUndercountVerify);
    const applyUndercountVerifyClean = !!(trackerParams && trackerParams.applyUndercountVerifyClean);
    const applyMotionVerify = !!(trackerParams && trackerParams.applyMotionVerify);
    const applyVarianceInflation = !!(trackerParams && trackerParams.applyVarianceInflation);
    const applyInterpolation = !!(trackerParams && trackerParams.applyInterpolation);
    const interpMaxGap = (trackerParams && trackerParams.interpMaxGap != null) ? trackerParams.interpMaxGap : 5;
    // Temporal 3D PAF flags (Landings 1-3)
    const applyTemporalPaf = !!(trackerParams && trackerParams.applyTemporalPaf);
    const pafLandings = (trackerParams && trackerParams.pafLandings != null) ? trackerParams.pafLandings : 3;
    if (applyKalman) console.error(`[luc3d] kalman ON (alpha=${kalmanAlpha})`);
    if (applyStitching) console.error(`[luc3d] stitching ON (maxGap=${stitchMaxGap})`);
    if (applyUndercountVerify) console.error(`[luc3d] undercount-verify ON (naive)`);
    if (applyUndercountVerifyClean) console.error(`[luc3d] undercount-verify ON (clean triangulation)`);
    if (applyMotionVerify) console.error(`[luc3d] motion-verify ON (SORT-style 3D motion tracker)`);
    if (applyVarianceInflation) console.error(`[luc3d] variance-inflation ON (soft propagation block)`);
    if (applyInterpolation) console.error(`[luc3d] interpolation ON (maxGap=${interpMaxGap})`);
    if (applyTemporalPaf) console.error(`[luc3d] temporal-3D-PAF ON (Landings 1-${pafLandings})`);
    let undercountStats = { camsVerified: 0, instancesReassigned: 0 };
    let viStats = { checked: 0, dropped: 0 };
    let pafStats = { kfMatched: 0, l2Swaps: 0, l3Flips: 0 };
    const motionTracker = applyMotionVerify ? sandbox.makeMotionTracker(trackerParams) : null;
    const tempPafKF = applyTemporalPaf ? new sandbox.KalmanIdentityTracker({
        qPos: (trackerParams && trackerParams.kfQPos) || 25.0,
        qVel: (trackerParams && trackerParams.kfQVel) || 9.0,
        rObs: (trackerParams && trackerParams.kfRObs) || 100.0,
        gate: (trackerParams && trackerParams.kfGate) || 16.0,
        maxAge: (trackerParams && trackerParams.kfMaxAge) || 30,
    }) : null;

    const t0 = Date.now();
    let prevAssignments = null, prevTargets3d = null;
    const kalmanState = applyKalman ? makeKalmanState() : null;
    const perFrame = [];
    // Landing 3 — one-frame buffer for forward lookahead. Each entry is the
    // FINAL output for a frame plus the targets3d we'd feed into the KF if
    // the frame stays as-is. We commit the buffered frame on the NEXT
    // iteration, after we've seen the next frame's evidence.
    let pafBuffered = null;
    for (const fi of session.frameIndices) {
        const fg = session.getFrameGroup(fi);
        const predictedPrev = applyKalman ? kalmanPredictTargets(prevTargets3d, kalmanState, fi, kalmanAlpha) : prevTargets3d;
        let r;
        try {
            r = matchFrameInstances(fg, session.cameras, session, {
                numAnimals: opts.numAnimals,
                perFrame: true,
                prevAssignments,
                prevTargets3d: predictedPrev,
                // Temporal 3D PAF — Landing 1: Mahalanobis innovation
                // added to per-frame Hungarian cost (Signal 5).
                kf: tempPafKF,
                frameIdx: fi,
            });
        } catch (e) {
            console.error(`[luc3d] frame ${fi} error: ${e.message}`);
            continue;
        }
        // Temporal 3D PAF — Landing 2: hypothesis-swap rescue. If any
        // group's KF Mahalanobis exceeds the gate, try permutations and pick
        // the lowest-total-Mahalanobis assignment.
        if (applyTemporalPaf && pafLandings >= 2 && opts.numAnimals) {
            const rescued = rescueByHypothesis(sandbox, r, tempPafKF, fi, opts.numAnimals, trackerParams);
            if (rescued.swapped) {
                pafStats.l2Swaps++;
                r.assignments = rescued.assignments;
                r.targets3d = rescued.targets3d;
            }
        }
        // Undercount verification: re-assign instances in under-counted cameras
        // using 3D triangulated identities. Three variants:
        //   - "motion": SORT-style 3D motion tracker. Maintains per-identity
        //     centroid+velocity, predicts position into under-counted cams.
        //     Handles short occlusions where the clean variant has no data.
        //   - "clean": Re-triangulates per identity from ONLY fully-covered
        //     cameras each frame. No motion model, but no contamination either.
        //   - "naive": Uses the tracker's own targets3d (may be contaminated).
        let assignmentsForOutput = r.assignments;
        if (applyMotionVerify && opts.numAnimals) {
            const fixed = sandbox.verifyUnderCountedCamerasMotion(fg, session.cameras,
                                                                  r.assignments,
                                                                  opts.numAnimals,
                                                                  motionTracker, fi,
                                                                  trackerParams);
            if (fixed && fixed.__stats) {
                undercountStats.camsVerified += fixed.__stats.camsVerified;
                undercountStats.instancesReassigned += fixed.__stats.instancesReassigned;
            }
            assignmentsForOutput = fixed || r.assignments;
        } else if (applyUndercountVerifyClean && opts.numAnimals) {
            const fixed = sandbox.verifyUnderCountedCamerasClean(fg, session.cameras,
                                                                 r.assignments,
                                                                 opts.numAnimals,
                                                                 trackerParams);
            if (fixed && fixed.__stats) {
                undercountStats.camsVerified += fixed.__stats.camsVerified;
                undercountStats.instancesReassigned += fixed.__stats.instancesReassigned;
            }
            assignmentsForOutput = fixed || r.assignments;
        } else if (applyUndercountVerify && r.targets3d && r.targets3d.length > 0 && opts.numAnimals) {
            const fixed = sandbox.verifyUnderCountedCameras(fg, session.cameras, /*camMap*/null,
                                                            r.targets3d, r.assignments,
                                                            opts.numAnimals,
                                                            trackerParams);
            if (fixed && fixed.__stats) {
                undercountStats.camsVerified += fixed.__stats.camsVerified;
                undercountStats.instancesReassigned += fixed.__stats.instancesReassigned;
            }
            assignmentsForOutput = fixed || r.assignments;
        }
        // Compute what to feed forward as prevAssignments. If variance-inflation
        // is on, this may DROP suspect entries (without changing the output).
        let nextPrev = assignmentsForOutput;
        if (applyVarianceInflation && opts.numAnimals) {
            const filtered = sandbox.filterInconsistentAssignments(fg, session.cameras,
                                                                   assignmentsForOutput,
                                                                   opts.numAnimals,
                                                                   trackerParams);
            if (filtered && filtered.__stats) {
                viStats.checked += filtered.__stats.checked;
                viStats.dropped += filtered.__stats.dropped;
            }
            if (filtered) nextPrev = filtered;
        }
        if (nextPrev && nextPrev.size > 0) prevAssignments = nextPrev;
        if (r.targets3d && r.targets3d.length > 0) prevTargets3d = r.targets3d;
        if (applyKalman) kalmanUpdate(r.targets3d, kalmanState, fi);

        // Build the per-frame record (assignments+targets3d we'd commit
        // for this frame if it stays as-is).
        const thisRecord = {
            frame: fi,
            assignments: assignmentsForOutput,        // Map, will be flattened on commit
            targets3d: r.targets3d.map(t => ({
                identityId: t.identityId ?? null,
                points3d: t.points3d,
                prevInstances: t.prevInstances,        // keep for lookahead flip
            })),
        };

        if (applyTemporalPaf && pafLandings >= 3) {
            // Landing 3 — one-frame buffer. Finalize the PREVIOUS frame's
            // record using the CURRENT frame's targets3d as forward evidence.
            // Then update KF with the (now-finalized) previous frame and
            // buffer the current frame for next iteration.
            if (pafBuffered) {
                const la = lookaheadFlip(sandbox, pafBuffered, r.targets3d, tempPafKF, fi, trackerParams);
                const committed = la.record;
                if (la.flipped) pafStats.l3Flips++;
                // Now commit `committed` to perFrame + update KF with its centroids.
                perFrame.push({
                    frame: committed.frame,
                    assignments: Array.from(committed.assignments.entries()),
                    targets3d: committed.targets3d.map(t => ({
                        identityId: t.identityId ?? null,
                        points3d: t.points3d,
                    })),
                });
                // KF update on the finalized previous frame
                for (const t of committed.targets3d) {
                    if (t.identityId == null || !t.points3d) continue;
                    const c = sandbox.computeCentroidFromPoints3d(t.points3d);
                    if (c) {
                        tempPafKF.update(t.identityId, c, committed.frame);
                        pafStats.kfMatched++;
                    }
                }
            }
            pafBuffered = thisRecord;
        } else {
            // Landings 1-2 or PAF off — commit this frame immediately.
            perFrame.push({
                frame: fi,
                assignments: Array.from(assignmentsForOutput.entries()),
                targets3d: r.targets3d.map(t => ({
                    identityId: t.identityId ?? null,
                    points3d: t.points3d,
                })),
            });
            if (applyTemporalPaf && tempPafKF) {
                // KF update immediately for Landings 1-2.
                for (const t of r.targets3d) {
                    if (t.identityId == null || !t.points3d) continue;
                    const c = sandbox.computeCentroidFromPoints3d(t.points3d);
                    if (c) {
                        tempPafKF.update(t.identityId, c, fi);
                        pafStats.kfMatched++;
                    }
                }
            }
        }
    }
    // Flush the Landing-3 buffer (last frame).
    if (pafBuffered) {
        perFrame.push({
            frame: pafBuffered.frame,
            assignments: Array.from(pafBuffered.assignments.entries()),
            targets3d: pafBuffered.targets3d.map(t => ({
                identityId: t.identityId ?? null,
                points3d: t.points3d,
            })),
        });
    }
    if (applyUndercountVerify || applyUndercountVerifyClean || applyMotionVerify) {
        console.error(`[luc3d] undercount: ${undercountStats.instancesReassigned} reassignments across ${undercountStats.camsVerified} cam-frames`);
    }
    if (applyVarianceInflation) {
        console.error(`[luc3d] variance-inflation: dropped ${viStats.dropped}/${viStats.checked} suspect votes from prevAssignments`);
    }
    if (applyTemporalPaf) {
        console.error(`[luc3d] temporal-3D-PAF: ${pafStats.kfMatched} KF updates, ${pafStats.l2Swaps} L2 hypothesis swaps, ${pafStats.l3Flips} L3 retroactive flips`);
    }
    if (applyStitching) stitchPerFrame(perFrame, stitchMaxGap);
    if (applyInterpolation) interpolateAssignments(perFrame, interpMaxGap);
    const elapsed = (Date.now() - t0) / 1000;
    console.error(`[luc3d] tracked ${perFrame.length} frames in ${elapsed.toFixed(1)}s (${(perFrame.length / Math.max(elapsed, 1e-6)).toFixed(1)} fps)`);

    // 6. Write output JSON
    fs.mkdirSync(path.dirname(opts.out), { recursive: true });
    fs.writeFileSync(opts.out, JSON.stringify({
        sessionIdx: opts.sessionIdx,
        numAnimals: opts.numAnimals,
        cameras,
        identities: session.identities.map(id => ({ id: id.id, name: id.name })),
        frames: perFrame,
        runtimeSeconds: elapsed,
    }));
    console.error(`[luc3d] wrote ${opts.out}`);
}

main().catch(e => { console.error(e); process.exit(1); });
