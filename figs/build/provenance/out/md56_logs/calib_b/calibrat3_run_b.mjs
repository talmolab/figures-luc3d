/**
 * calibrat3 end to end in headless Chromium -- figs/fig7_calib_calibrat3.mjs, instrumented
 * for the apples-to-apples rerun. The app, its defaults and the clicks are UNCHANGED.
 *
 * Differences from fig7_calib_calibrat3.mjs, all budget or bookkeeping:
 *   * HW=<n> overrides navigator.hardwareConcurrency BEFORE the app loads, so calibrat3
 *     sizes its pools from the same core budget Anipose is given (detect pool =
 *     min(12, HW-2), calib pool = min(4, HW-2); loading/detect-pool.js, calib-client.js).
 *     Chromium does not read the taskset affinity for this, so without the override it
 *     would report 64 on this box regardless of the cpuset.
 *   * records Chromium/Playwright versions, the pool sizes the app actually logged, and
 *     the frame stride, into <OUT>.timing.json.
 *
 *   SESSION=/dir OUT=/prefix [TARGET=800|ALL_FRAMES=1] [REF_CAM=Camera6] [HW=16]
 *   [BASE=http://localhost:8093] node calibrat3_run_b.mjs
 */
import { chromium } from 'playwright';
import { writeFileSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const BASE = process.env.BASE || 'http://localhost:8093';
const SESSION = process.env.SESSION, OUT = process.env.OUT;
if (!SESSION || !OUT) { console.error('SESSION= and OUT= are required'); process.exit(2); }
const ALL_FRAMES = process.env.ALL_FRAMES === '1';
const TARGET = process.env.TARGET || '800';
const REF_CAM = process.env.REF_CAM || null;
const HW = process.env.HW ? Number(process.env.HW) : null;
let pwVersion = null;
try { pwVersion = JSON.parse(readFileSync(createRequire(import.meta.url).resolve('playwright/package.json'))).version; } catch { /* recorded as null */ }

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1700, height: 1050 } });
if (HW) await page.addInitScript((n) => { Object.defineProperty(Navigator.prototype, 'hardwareConcurrency', { get: () => n, configurable: true }); }, HW);
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
page.on('console', (m) => { if (m.type() === 'error') errors.push(`console.error: ${m.text()}`); });
const t0 = Date.now();
const step = (msg) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s] ${msg}`);
const waitState = (pred, arg, opts) => Promise.race([
    page.waitForFunction(pred, arg, opts),
    (async () => { for (;;) { await page.waitForTimeout(1000); const b = await page.evaluate(() => { const e = document.getElementById('errorMsg'); return e && e.style.display === 'block' ? e.textContent : null; }).catch(() => null); if (b) throw new Error(`app error banner: ${b}`); } })(),
]);
const timing = {};
const logLines = () => page.evaluate(() => Array.from(document.querySelectorAll('#logBody .log-entry')).map(x => x.textContent)).catch(() => []);

try {
    await page.goto(`${BASE}/index.html`, { waitUntil: 'load' });
    await waitState(() => document.getElementById('workerStatus').classList.contains('ok'), null, { timeout: 300000 });
    const hwSeen = await page.evaluate(() => navigator.hardwareConcurrency);

    const tL = Date.now();
    await page.setInputFiles('#folderInput', SESSION, { timeout: 600000 });
    await waitState(() => window.__calibrat3.state.views.length >= 2 && window.__calibrat3.state.totalFrames > 0, null, { timeout: 900000 });
    timing.load_s = (Date.now() - tL) / 1000;
    const info = await page.evaluate(() => ({
        layout: window.__calibrat3.state.sessionLayout,
        views: window.__calibrat3.state.views.map(v => v.name),
        frames: window.__calibrat3.state.totalFrames,
        board: window.__calibrat3.state.board,
        size: [window.__calibrat3.state.views[0].info.width, window.__calibrat3.state.views[0].info.height],
    }));
    step(`session: ${info.views.length} views ${info.size.join('x')}, ${info.frames} frames, layout ${info.layout}, board ${JSON.stringify(info.board)}; hardwareConcurrency seen ${hwSeen}`);

    if (ALL_FRAMES) { await page.check('#allFramesCheck'); await page.dispatchEvent('#allFramesCheck', 'change'); }
    else { await page.fill('#targetSamples', TARGET); await page.dispatchEvent('#targetSamples', 'input'); }
    const tD = Date.now();
    await page.click('#runDetectionBtn');
    await waitState(() => !window.__calibrat3.state.detectionRunning && window.__calibrat3.state.detections && window.__calibrat3.state.detections.size > 0, null, { timeout: 7200000 });
    timing.detection_s = (Date.now() - tD) / 1000;
    const det = await page.evaluate(() => window.__calibrat3.state.detections.summary(6));
    step(`detection: ${det.frames} frames x ${info.views.length} views in ${timing.detection_s.toFixed(1)} s; per view ${JSON.stringify(det.perView)}`);

    const tI = Date.now();
    await page.click('#computeIntrinsicsBtn');
    await waitState(() => document.getElementById('computeIntrinsicsBtn').disabled === false && window.__calibrat3.state.intrinsics.length > 0, null, { timeout: 7200000 });
    timing.intrinsics_s = (Date.now() - tI) / 1000;
    step(`intrinsics in ${timing.intrinsics_s.toFixed(1)} s`);

    if (REF_CAM !== null) {
        const idx = info.views.findIndex(n => n === REF_CAM || new RegExp(`(^|[-_ ])${REF_CAM}([-_ .]|$)`).test(n));
        if (idx < 0) throw new Error(`REF_CAM ${REF_CAM} not among ${info.views.join(', ')}`);
        await page.selectOption('#referenceCamera', String(idx));
        timing.reference_camera = info.views[idx];
    }
    const tE = Date.now();
    await page.click('#computeExtrinsicsBtn');
    await waitState(() => window.__calibrat3.state.reproj !== null, null, { timeout: 7200000 });
    timing.extrinsics_s = (Date.now() - tE) / 1000;
    const ext0 = await page.evaluate(() => window.__calibrat3.state.reproj.summary);
    timing.reference_camera_used = await page.evaluate(() => { const s = window.__calibrat3.state; const i = Number(document.getElementById('referenceCamera').value); return s.views[i] ? s.views[i].name : null; });
    step(`extrinsics in ${timing.extrinsics_s.toFixed(1)} s (reference ${timing.reference_camera_used}): median ${ext0.overall.median.toFixed(2)} px`);

    await page.waitForSelector('#runSbaBtn:not([disabled])', { timeout: 120000 });
    const tS = Date.now();
    await page.click('#runSbaBtn');
    await waitState(() => window.__calibrat3.state.sbaResult !== null, null, { timeout: 10800000 });
    await page.waitForSelector('#runSbaBtn:not([disabled])', { timeout: 10800000 });
    timing.sba_s = (Date.now() - tS) / 1000;
    const ext1 = await page.evaluate(() => window.__calibrat3.state.reproj.summary);
    step(`SBA in ${timing.sba_s.toFixed(1)} s: median ${ext1.overall.median.toFixed(2)} px p95 ${ext1.overall.p95.toFixed(1)}`);

    const toml = await page.evaluate(async () => (await import('./ui/stage-export.js')).buildToml());
    writeFileSync(`${OUT}.toml`, toml);
    const json = await page.evaluate(async () => { const { serializeSession } = await import('./import-export/session-save.js'); return JSON.stringify(serializeSession(window.__calibrat3.state)); });
    writeFileSync(`${OUT}.session.json`, json);
    timing.total_s = (Date.now() - t0) / 1000;
    const pools = (await logLines()).filter(l => /pool ready/i.test(l));
    writeFileSync(`${OUT}.timing.json`, JSON.stringify({
        session: SESSION, allFrames: ALL_FRAMES, target: ALL_FRAMES ? null : Number(TARGET),
        views: info.views, size: info.size, totalFrames: info.frames,
        detectionFrames: det.frames, detectionPerView: det.perView, timing,
        appSummary: { initial: ext0.overall, refined: ext1.overall },
        budget: { hardwareConcurrencyOverride: HW, hardwareConcurrencySeen: hwSeen, poolLog: pools },
        versions: { chromium: browser.version(), playwright: pwVersion, node: process.version },
    }, null, 1));
    step(`wrote ${OUT}.{toml,session.json,timing.json}`);
} catch (e) {
    errors.push(`driver: ${e.message}`);
    console.log('--- app log tail ---\n' + (await logLines()).slice(-20).join('\n'));
} finally {
    await browser.close();
}
if (errors.length) { console.log('\nFAILED:\n' + errors.join('\n')); process.exit(1); }
console.log('\nDONE');
