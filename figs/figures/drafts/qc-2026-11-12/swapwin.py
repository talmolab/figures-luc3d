import h5py, numpy as np, glob
s='20250904_183921'; c='21369048'
with h5py.File(f'/root/vast/eric/luc3d-bench/outputs/bmimica/gt/{s}/{c}/proofread.analysis.h5','r') as h: g=np.transpose(h['tracks'][:],(3,0,2,1))
a=glob.glob(f'/root/vast/eric/BMimica/{s}/{c}/*.analysis.h5')[0]
with h5py.File(a,'r') as h: r=np.transpose(h['tracks'][:],(3,0,2,1)); print(dict(h.attrs) if h.attrs else '', [k for k in h.keys()]); prov=h['provenance'][()] if 'provenance' in h else None; lp=h['labels_path'][()] if 'labels_path' in h else None
print('labels_path',lp); print('provenance',str(prov)[:400])
n=min(len(g),len(r))
with np.errstate(all='ignore'):
    d00=np.nanmedian(np.linalg.norm(r[:n,0]-g[:n,0],axis=-1),-1); d01=np.nanmedian(np.linalg.norm(r[:n,0]-g[:n,1],axis=-1),-1)
sw=(d01<d00)
for a_,b_ in [(0,20000),(20000,40000),(40000,45000),(45000,60000),(60000,80000),(80000,100000),(100000,115300),(115300,130000),(130000,180058)]:
    print(f'{a_:6d}-{b_:6d}: slot0->animal1 share {sw[a_:b_].mean():.2f}')
idx=np.where(sw)[0]; print('first swapped frame',idx.min(),'last',idx.max())
