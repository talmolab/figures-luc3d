import json, numpy as np, pandas as pd, os
from scipy import stats
os.chdir('/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs')
FL=['20250904_183921','20250905_203607','20250904_140306']
SETS={'all':[], 'ex183921':[FL[0]], 'ex203607':[FL[1]], 'ex140306':[FL[2]], 'ex_all3':FL}
U=json.load(open('out/fig5_upright.json'))
ld=pd.read_csv('data/fig5/fig5f_leader_by_session.csv'); ld=ld[ld.session.str.match(r'^\d{8}_\d{6}$')].set_index('session')
ps={p['session']:p for p in U['per_session']}
e0=U['events'][0]; print('initiator_track example',e0['initiator_track'], 'taller_track',e0['taller_track'])
rows=[]
for e in U['events']:
    s=e['session']; male=str(int(ld.loc[s,'male_id'])); pt=ps[s]['per_track']
    mi=[i for i,t in enumerate(pt) if str(t['animal'])==male][0]
    it=e['initiator_track']
    if isinstance(it,str): it=[i for i,t in enumerate(pt) if t['track_name']==it or str(t['animal'])==it]; it=it[0] if it else None
    rows.append(dict(session=s,pair=f"{male}|{int(ld.loc[s,'female_id'])}",male=male,female=str(int(ld.loc[s,'female_id'])),
        init=None if it is None else ('male' if it==mi else 'female'), lag=e['lag_s'],
        sp_m=e[f'speed_bl_s_t{mi}'], sp_f=e[f'speed_bl_s_t{1-mi}'], pu_m=e[f'pursuit_rel_t{mi}'], pu_f=e[f'pursuit_rel_t{1-mi}']))
E=pd.DataFrame(rows)
for name,ex in SETS.items():
    D=E[~E.session.isin(ex)]
    fl=D[D.init=='female']
    S=D.groupby('session').agg(m=('sp_m','median'),f=('sp_f','median'),pm=('pu_m','median'),pf=('pu_f','median'),n=('lag','size'),pair=('pair','first'))
    P=D.groupby('pair').agg(m=('sp_m','median'),f=('sp_f','median'),pm=('pu_m','median'),pf=('pu_f','median'),n=('lag','size')); P=P[P.n>=5]
    Am=D.groupby('male').agg(sp=('sp_m','median'),pu=('pu_m','median')); Af=D.groupby('female').agg(sp=('sp_f','median'),pu=('pu_f','median'))
    print(f'### {name}: female-led n={len(fl)} median lag {fl.lag.median():.2f}s; session check speed {S.m.median():.2f}/{S.f.median():.2f}; '
          f'pairs(>=5) n={len(P)} male faster {(P.m>P.f).sum()}, pursuit lower {(P.pm<P.pf).sum()}; animals {len(Am)}M/{len(Af)}F '
          f'MW pursuit P={stats.mannwhitneyu(Am.pu,Af.pu).pvalue:.3f}, every male < every female: {Am.pu.max()<Af.pu.min()}; MW speed P={stats.mannwhitneyu(Am.sp,Af.sp).pvalue:.2f}')
b=json.load(open('out/fig5_rear_coupling_2animal_perdir.json'))['corpora']['BMimica']
for k in ['far_i0','far_i1']:
    print(k,'sessions:',[s for s in FL if s in b[k]['sessions']], 'n',b[k]['n_sessions'])
