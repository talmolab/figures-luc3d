import h5py, numpy as np, glob, itertools
s='20250904_140306'
for c in ["21241563","21369048","21372315","21372316","22085397"]:
    with h5py.File(f'/root/vast/eric/luc3d-bench/outputs/bmimica/gt/{s}/{c}/proofread.analysis.h5','r') as h: g=np.transpose(h['tracks'][:],(3,0,2,1))
    a=glob.glob(f'/root/vast/eric/BMimica/{s}/{c}/*.analysis.h5')[0]
    with h5py.File(a,'r') as h: r=np.transpose(h['tracks'][:],(3,0,2,1))
    fidx=np.arange(0,min(len(g),len(r)),15); r=r[fidx]; g=g[fidx]; T=r.shape[1]
    cost=np.full((len(fidx),2,T),np.inf)
    for aa in range(2):
        for tt in range(T):
            with np.errstate(all='ignore'): cost[:,aa,tt]=np.nanmean(np.linalg.norm(r[:,tt]-g[:,aa],axis=-1),axis=1)
    cost=np.nan_to_num(cost,nan=np.inf,posinf=np.inf)
    perms=list(itertools.permutations(range(T),2))
    tot=np.stack([cost[:,0,p[0]]+cost[:,1,p[1]] for p in perms],1)
    pick=np.argmin(tot,1)
    ok=np.array([cost[i,0,perms[pick[i]][0]]<60 and cost[i,1,perms[pick[i]][1]]<60 for i in range(len(fidx))])
    # frames where tot is inf (a perm with a NaN track chosen)
    print(c,'T',T,'frames with both animals matched <60px: %.3f'%ok.mean(), ' frames where best perm cost is inf: %.3f'%np.isinf(tot.min(1)).mean())
