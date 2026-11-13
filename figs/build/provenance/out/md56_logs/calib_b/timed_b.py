"""Run one timed command inside a fixed cpuset and log the machine around it.

    python3 timed_b.py <out.json> <cpulist e.g. 0-15> -- <command ...>

1. QUIET GATE: waits (up to QUIET_TIMEOUT s, default 1800) until the WHOLE machine has
   averaged < QUIET_CORES busy cores (default 1.5) over a 10 s window, then starts. If the
   gate times out the run still happens but is flagged `quiet_gate_passed: false`.
2. Runs the command under `taskset -c <cpulist>` (children inherit the affinity:
   Pool workers, Chromium and all its renderer/GPU/utility processes).
3. Samples /proc/stat once a second: busy cores INSIDE the cpuset (the tool) and OUTSIDE
   it (anything else on the box -- the contamination the old "idle machine" claim never
   measured). Records /proc/loadavg, wall time and the children's user+sys CPU.
"""
import sys, os, json, time, subprocess, resource, socket, threading

out_json, cpulist = sys.argv[1], sys.argv[2]
cmd = sys.argv[sys.argv.index('--') + 1:]
QUIET_CORES = float(os.environ.get('QUIET_CORES', '1.5'))
QUIET_TIMEOUT = float(os.environ.get('QUIET_TIMEOUT', '1800'))


def parse_cpulist(s):
    cpus = set()
    for part in s.split(','):
        a, _, b = part.partition('-')
        cpus.update(range(int(a), int(b or a) + 1))
    return cpus


INSIDE = parse_cpulist(cpulist)


def snap():
    per = {}
    for line in open('/proc/stat'):
        if line.startswith('cpu') and line[3].isdigit():
            f = line.split(); v = list(map(int, f[1:9]))
            idle = v[3] + v[4]
            per[int(f[0][3:])] = (sum(v) - idle, sum(v))
    return per


def busy(a, b):
    """busy cores inside / outside the cpuset between two snapshots."""
    ins = outs = 0.0
    for c in b:
        db, dt = b[c][0] - a[c][0], b[c][1] - a[c][1]
        frac = db / dt if dt else 0.0
        if c in INSIDE: ins += frac
        else: outs += frac
    return ins, outs


# ---- 1. quiet gate
t_gate = time.time(); passed = False; window = []
prev = snap()
while time.time() - t_gate < QUIET_TIMEOUT:
    time.sleep(1); cur = snap(); i, o = busy(prev, cur); prev = cur
    window = (window + [i + o])[-10:]
    if len(window) == 10 and sum(window) / 10 < QUIET_CORES:
        passed = True; break
gate = dict(quiet_gate_passed=passed, quiet_wait_s=time.time() - t_gate,
            busy_cores_before_start=sum(window) / max(1, len(window)), threshold_cores=QUIET_CORES)
print(f'[timed_b] quiet gate {"passed" if passed else "TIMED OUT"} after {gate["quiet_wait_s"]:.0f} s '
      f'(machine busy {gate["busy_cores_before_start"]:.2f} cores)', flush=True)

# ---- 2/3. run + sample
samples, stop = [], threading.Event()


def sampler():
    p = snap(); t = time.time()
    while not stop.wait(1.0):
        c = snap(); i, o = busy(p, c); p = c
        samples.append((round(time.time() - t, 1), round(i, 2), round(o, 2)))


ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
load0 = open('/proc/loadavg').read().split()[:3]
th = threading.Thread(target=sampler, daemon=True); th.start()
t0 = time.time()
rc = subprocess.call(['taskset', '-c', cpulist] + cmd)
wall = time.time() - t0
stop.set(); th.join()
ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
outside = [s[2] for s in samples] or [0.0]
inside = [s[1] for s in samples] or [0.0]
rec = dict(cmd=cmd, cpulist=cpulist, n_cpus=len(INSIDE), host=socket.gethostname(),
           nproc=os.cpu_count(), started_unix=t0, wall_s=wall, returncode=rc,
           children_cpu_s=(ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime),
           loadavg_start=load0, loadavg_end=open('/proc/loadavg').read().split()[:3],
           inside_busy_cores_mean=sum(inside) / len(inside), inside_busy_cores_max=max(inside),
           outside_busy_cores_mean=sum(outside) / len(outside), outside_busy_cores_max=max(outside),
           samples=samples, **gate)
json.dump(rec, open(out_json, 'w'), indent=1)
print(f'[timed_b] rc={rc} wall {wall:.1f} s, CPU {rec["children_cpu_s"]:.0f} s, '
      f'outside-cpuset load mean {rec["outside_busy_cores_mean"]:.2f} / max {rec["outside_busy_cores_max"]:.2f} cores', flush=True)
sys.exit(rc)
