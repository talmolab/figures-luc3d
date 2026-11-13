import json, numpy as np, os
os.chdir('/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs/out')
FL=['20250904_183921','20250905_203607','20250904_140306']
SETS={'all':[], 'ex183921':[FL[0]], 'ex203607':[FL[1]], 'ex140306':[FL[2]], 'ex_all3':FL}
def rank(vals, s, key):
    v=sorted(vals.items(), key=lambda kv: kv[1]); return [k for k,_ in v].index(s)+1
# --- headline tracker (fig8_methods_50_b3, sync_stale20_dist25)
o=json.load(open('fig8_methods_50_b3.json')); cf=o['camera_frames_by_session']
c=[c for c in o['cells'] if c['config']=='sync_stale20_dist25'][0]
ps={p['session']:p for p in c['per_session']}
print('fig8 stale20 deposit: idf1_cross',c['idf1_cross'],'switches',c['switches'],'per100k',c['switches_per_100k_camera_frames'], 'n',len(ps))
for s in FL:
    p=ps[s]; cr={k:v['cross_idf1'] for k,v in ps.items()}
    print(f'  {s}: cross_idf1={p["cross_idf1"]:.4f} within={p["within_idf1"]:.4f} cross_sw={p["cross_switches"]} within_sw={p["within_switches"]}  rank(low->high cross idf1)={rank(cr,s,0)}/56')
for name,ex in SETS.items():
    k=[s for s in ps if s not in ex]
    cr=np.array([ps[s]['cross_idf1'] for s in k]); wi=np.array([ps[s]['within_idf1'] for s in k])
    sw=sum(ps[s]['within_switches'] for s in k); swc=sum(ps[s]['cross_switches'] for s in k); fr=sum(sum(cf[s].values()) for s in k)
    print(f'  {name:10s} n={len(k)} mean_cross={cr.mean():.4f} median_cross={np.median(cr):.4f} mean_within={wi.mean():.4f} within_sw={sw} cross_sw={swc} per100k(within)={sw/fr*1e5:.3f} per100k(cross)={swc/fr*1e5:.3f}')
# --- Fig 2H/I sweep, r=6, framematched
o=json.load(open('fig3_sweep50__distanceThreshold25-stale20-sync_e508a7ab_framematched_b3.json'))
for r in [0,6]:
    c=[c for c in o['cells'] if c['corr3d']==r][0]; ps={p['session']:p for p in c['per_session']}
    print(f'sweep r={r} deposit per100k={c["switches_per_100k_camera_frames"]:.3f} idf1={c["idf1_cross"]:.4f} switches={c["switches"]}')
    for s in FL: print(f'   {s}: cross_idf1={ps[s]["cross_idf1"]:.4f} within_sw={ps[s]["within_switches"]} cross_sw={ps[s]["cross_switches"]} camfr={ps[s]["camera_frames"]}')
    for name,ex in SETS.items():
        k=[s for s in ps if s not in ex]
        cr=np.mean([ps[s]['cross_idf1'] for s in k]); sw=sum(ps[s]['within_switches'] for s in k); swc=sum(ps[s]['cross_switches'] for s in k); fr=sum(ps[s]['camera_frames'] for s in k)
        print(f'   {name:10s} mean_cross={cr:.4f} per100k(within)={sw/fr*1e5:.3f} per100k(cross)={swc/fr*1e5:.3f}')
o=json.load(open('fig3_frame_matched_bmimica_b3.json')); ps={p['session']:p for p in o['per_session']}
print('frame-matched exhaustive arm deposit:', {k:v for k,v in o['arms']['exhaustive'].items() if not isinstance(v,(dict,list))})
for s in FL: print('  ',s, {k:ps[s][k] for k in ['exhaustive_idf1_cross','exhaustive_switches','frames_scored']})
for name,ex in SETS.items():
    k=[s for s in ps if s not in ex]
    print(f'   {name:10s} exh mean_cross={np.mean([ps[s]["exhaustive_idf1_cross"] for s in k]):.4f} exh switches={sum(ps[s]["exhaustive_switches"] for s in k)} per100k={sum(ps[s]["exhaustive_switches"] for s in k)/sum(ps[s]["frames_scored"]*5 for s in k)*1e5:.3f}')
# --- Fig 2F quality (MD config)
for f,arm in [('fig3_quality_b3.json','exhaustive'),('md56main/fig3_quality__distanceThreshold25-stale20_f7733dbf_b3.json','greedy')]:
    o=json.load(open(f)); cfg=[c for c in o['configs'] if 'bmimica' in c['key'].lower() or c['dataset'].lower().startswith(('bm','mouse'))][0]
    ps={p['session']:p for p in cfg['per_session']}
    rate={s:100*(1-p['gt'][arm]['exact_match_frames']/p['gt'][arm]['frames']) for s,p in ps.items() if p['gt'][arm]['frames']}
    print(f'Fig2F {arm} ({cfg["key"]}) n={len(rate)}')
    for s in FL: print(f'   {s}: misgrouped%={rate.get(s)}  rank(high=worst)={56-rank(rate,s,0)+1 if s in rate else None}')
    for name,ex in SETS.items():
        k=[s for s in rate if s not in ex]; v=np.array([rate[s] for s in k])
        pooled=100*sum(ps[s]['gt'][arm]['frames']-ps[s]['gt'][arm]['exact_match_frames'] for s in k)/sum(ps[s]['gt'][arm]['frames'] for s in k)
        print(f'   {name:10s} median={np.median(v):.4f}% max={v.max():.3f}% pooled_MD={pooled:.4f}% perfect={int((v==0).sum())}')
# --- Supp 4A baselines
for f,arm,key in [('fig7_sleap_max2_c6b3nd.json',None,'sleap'),('fig7_bytetrack_max2_c6b3nd.json','arm_B1s','bytetrack')]:
    o=json.load(open(f)); ps={p['session']:(p[arm] if arm else p) for p in o['per_session']}
    wk='within_idf1'; ck='cross_idf1_scoped'
    for s in FL: print(f'  {key} {s}: within={ps[s][wk]:.4f} cross={ps[s][ck]:.4f}')
    for name,ex in SETS.items():
        k=[s for s in ps if s not in ex]
        print(f'  {key} {name:10s} within mean={np.mean([ps[s][wk] for s in k]):.4f} cross mean={np.mean([ps[s][ck] for s in k]):.4f}')
# --- camera subsets
o=json.load(open('fig2_cams_identity_b3.json')); runs=o['per_run']
subs=sorted(set(r['subset'] for r in runs))
for name,ex in [('all',[]),('ex_all3',FL)]:
    out=[]
    for sb in subs:
        rr=[r for r in runs if r['subset']==sb and r['session'] not in ex]
        out.append(f"{sb}:sw={sum(r['within_switches'] for r in rr)},idf1={np.mean([r['within_idf1'] for r in rr]):.3f}")
    print('cams',name,' '.join(out))
for s in FL:
    print('  cams',s,' '.join(f"{r['subset']}:{r['within_switches']}" for r in runs if r['session']==s))
