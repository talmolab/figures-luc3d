import json, numpy as np, os, csv
os.chdir('/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs')
FL=['20250904_183921','20250905_203607','20250904_140306']
SETS={'all':[], 'ex183921':[FL[0]], 'ex203607':[FL[1]], 'ex140306':[FL[2]], 'ex_all3':FL}
def rank_desc(d,s): v=sorted(d,key=lambda k:-d[k]); return v.index(s)+1
o=json.load(open('out/fig4_by_views_js.json')); ps={p['session']:p for p in o['per_session']}
print('deposit err3d across sessions k2/k5:',o['err3d_mm_across_sessions']['2']['dlt']['p50'],o['err3d_mm_across_sessions']['5']['dlt']['p50'],' heldout k2/k4:',o['heldout_px_across_sessions']['2']['dlt']['p50'],o['heldout_px_across_sessions']['4']['dlt']['p50'])
print(' example heldout_acc_by_k keys', list(ps[FL[0]]['heldout_acc_by_k'].keys()), json.dumps(ps[FL[0]]['heldout_acc_by_k']['2'])[:200], ' heldout_px_by_k', json.dumps(ps[FL[0]]['heldout_px_by_k'].get('2'))[:200])
e5={s:p['by_k']['5']['dlt']['p50'] for s,p in ps.items()}; e2={s:p['by_k']['2']['dlt']['p50'] for s,p in ps.items()}
for s in FL: print(f'  {s}: err3d 5cam p50={e5[s]:.3f} mm (rank worst {rank_desc(e5,s)}/56), 2cam p50={e2[s]:.3f} (rank {rank_desc(e2,s)})')
print('  session distribution 5cam p50: median %.3f, p90 %.3f, max %.3f'%(np.median(list(e5.values())),np.percentile(list(e5.values()),90),max(e5.values())))
h2={s:p['heldout_px_by_k']['2']['dlt']['p50'] for s,p in ps.items()}; h4={s:p['heldout_px_by_k']['4']['dlt']['p50'] for s,p in ps.items()}
acc10={s:p['heldout_acc_by_k']['2']['dlt']['acc10'] for s,p in ps.items()}; acc5={s:p['heldout_acc_by_k']['2']['dlt']['acc5'] for s,p in ps.items()}
for name,ex in SETS.items():
    k=[s for s in ps if s not in ex]
    m10=np.median([acc10[s] for s in k]); m5=np.median([acc5[s] for s in k])
    print(f'  {name:9s} median-of-session err3d k2={np.median([e2[s] for s in k]):.3f} k5={np.median([e5[s] for s in k]):.3f} mm | heldout k2={np.median([h2[s] for s in k]):.3f} k4={np.median([h4[s] for s in k]):.3f} px | placements tau10={30+45*(1-m10):.2f} tau5={30+45*(1-m5):.2f}')
# pairs (Fig 3D)
pairs=list(ps[FL[0]]['pairs_dlt_err3d_mm'].keys()); print(' pair keys',pairs[:3], json.dumps(ps[FL[0]]['pairs_dlt_err3d_mm'][pairs[0]])[:200])
# Fig 3E bottom, 3F, 3G from CSVs
rows=list(csv.DictReader(open('data/fig3/fig3e_worst_camera.csv')))
W={r['session']:r for r in rows}
for s in FL: print(f'  3E {s}: all {float(W[s]["all_views_px"]):.3f} worst-dropped {float(W[s]["worst_dropped_px"]):.3f} refined {float(W[s]["refined_all_views_px"]):.3f}->{float(W[s]["refined_worst_dropped_px"]):.3f}')
for name,ex in SETS.items():
    k=[s for s in W if s not in ex]
    a=np.mean([float(W[s]['all_views_px']) for s in k]); b=np.mean([float(W[s]['worst_dropped_px']) for s in k])
    print(f'  3E {name:9s} DLT mean {a:.3f}->{b:.3f}; lower in {sum(float(W[s]["worst_dropped_px"])<float(W[s]["all_views_px"]) for s in k)}/{len(k)}')
rows=[r for r in csv.DictReader(open('data/fig3/fig3f_per_session.csv')) if r['group']=='reproj_p50']
F={r['session']:{k:float(r[k]) for k in ['anipose','dlt','anipose_optim','refined']} for r in rows}
all_med={k:np.median([F[s][k] for s in F]) for k in ['anipose','dlt','anipose_optim','refined']}
for s in FL: print('  3F',s,{k:round(v,3) for k,v in F[s].items()}, 'rank(dlt worst)',rank_desc({x:F[x]['dlt'] for x in F},s))
for name,ex in SETS.items():
    k=[s for s in F if s not in ex]
    print(f'  3F {name:9s}', {m:round(float(np.median([F[s][m] for s in k])),3) for m in ['refined','anipose_optim','anipose','dlt']}, 'refined lowest in', sum(F[s]['refined']<min(F[s]['anipose'],F[s]['dlt'],F[s]['anipose_optim']) for s in k),'/',len(k))
rows=list(csv.DictReader(open('data/fig3/fig3g_anipose_optim_accuracy.csv')))
for name,ex in [('all',[]),('ex_all3',FL)]:
    k=[r for r in rows if r['session'] not in ex]
    print('  3G smoothing worse in', sum(float(r['optim_heldout_p50'])>float(r['linear_heldout_p50']) for r in k),'/',len(k))
