import json, numpy as np, pandas as pd, os
from scipy import stats
os.chdir('/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs')
FL=['20250904_183921','20250905_203607','20250904_140306']
SETS={'all':[], 'ex183921':[FL[0]], 'ex203607':[FL[1]], 'ex140306':[FL[2]], 'ex_all3':FL}
sp=pd.read_csv('data/fig5/fig5d_speed_by_session.csv'); sp=sp[sp.session.str.match(r'^\d{8}_\d{6}$')]
pu=pd.read_csv('data/fig5/fig5e_pursuit_by_session.csv'); pu=pu[pu.session.str.match(r'^\d{8}_\d{6}$')]
ld=pd.read_csv('data/fig5/fig5f_leader_by_session.csv'); ld=ld[ld.session.str.match(r'^\d{8}_\d{6}$')]
ct=pd.read_csv('data/fig5/fig5s_controls.csv')
ev=json.load(open('out/fig5_upright.json'))['events']
def pairstat(df,mincount):
    g=df.groupby('pair').agg(n=('n_displays','sum'),m=('male_median','median'),f=('female_median','median'))
    return g
for name,ex in SETS.items():
    S=sp[~sp.session.isin(ex)]; P=pu[~pu.session.isin(ex)]; L=ld[~ld.session.isin(ex)]
    E=[e for e in ev if e['session'] not in ex]
    out=[]
    out.append(f'displays={len(E)} in {L.shape[0]} sessions; median dur={np.median([e["dur_s"] for e in E]):.2f}s')
    fl=[e for e in E if e.get('lag_s') is not None and e['initiator_track'] is not None]
    # female-led lag: need sex of initiator; use per-session leader table counts instead
    out.append(f'speed: male med {S.male_median.median():.2f} female {S.female_median.median():.2f}, male faster {(S.male_median>S.female_median).sum()}/{len(S)}, P={stats.wilcoxon(S.male_median,S.female_median).pvalue:.2g}')
    out.append(f'pursuit: male {P.male_median.median():.2f} female {P.female_median.median():.2f}, male lower {(P.male_median<P.female_median).sum()}/{len(P)}, P={stats.wilcoxon(P.male_median,P.female_median).pvalue:.2g}')
    # pair level (>=5 displays): mean? use per-pair median of session medians weighted? replicate: pair = median over the pair's displays unknown -> approximate with median of session medians
    fem=int(L.female_lead.sum()); mal=int(L.male_lead.sum())
    big=L[L.displays>=6]
    w=stats.wilcoxon(big.female_share-big.male_share).pvalue
    out.append(f'leader: female {fem}/{fem+mal} ({100*fem/(fem+mal):.1f}%), female-more in {(L.female_lead>L.male_lead).sum()}/{len(L)}; >=6 displays n={len(big)} median share {big.female_share.median():.2f} P={w:.2g}')
    # pairs
    L2=L.assign(pair=L.male_id.astype(str)+'|'+L.female_id.astype(str)).groupby('pair').agg(d=('displays','sum'),f=('female_lead','sum'),m=('male_lead','sum'))
    pb=L2[L2.d>=5]
    out.append(f'pairs: {len(L2)} pairings, female leads in {(L2.f>L2.m).sum()}; >=5: {len(pb)} pairs, {int(pb.d.sum())} displays, median female share {np.median(pb.f/pb.d):.2f}, leads {(pb.f>pb.m).sum()}/{len(pb)} sign P={stats.binomtest(int((pb.f>pb.m).sum()),len(pb)).pvalue:.2g}, pooled {int(pb.f.sum())}/{int(pb.f.sum()+pb.m.sum())}')
    # controls
    base=ct[(ct['mode']=='per_animal')&(~ct.session.isin(ex))]; bb=base[base.n>=6]
    fshare=bb.female_lead/(bb.female_lead+bb.male_lead); frear=bb.female_rear_frac/(bb.female_rear_frac+bb.male_rear_frac)
    r,p=stats.pearsonr(fshare,frear)
    ab=ct[(ct['mode']=='abs')].set_index('session')
    fem_abs=int((ab.loc[bb.session].leader=='female').sum())
    allb=base[base.n>=1]; flg=allb[allb.L_female_mm>allb.L_male_mm]; k=int(flg.female_lead.sum()); n=int(flg.female_lead.sum()+flg.male_lead.sum())
    out.append(f'controls: r={r:.3f} P={p:.2f} n={len(bb)}; abs60mm female leader {fem_abs}/{len(bb)}; female-longer sessions {len(flg)}: {k}/{n} ({100*k/n:.0f}%) P={stats.binomtest(k,n).pvalue:.1g}')
    print(f'### {name}'); [print('   ',x) for x in out]
# per-session contributions
for s in FL:
    r=ld[ld.session==s]
    print(s, 'displays',r.displays.values, 'female_lead',r.female_lead.values,'male_lead',r.male_lead.values)
