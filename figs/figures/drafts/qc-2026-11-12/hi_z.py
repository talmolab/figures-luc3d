import h5py, numpy as np, glob, sys
s=sys.argv[1]; f=glob.glob(f'/root/vast/eric/BMimica/{s}/*_points3d*.h5')[0]
with h5py.File(f,'r') as h:
    t=h['tracks'][:]; nodes=[n.decode() for n in h['node_names'][:]]
ix={n:i for i,n in enumerate(nodes)}
F=t.shape[0]
for a in range(2):
    z=t[:,a,:,2]; mz=np.median(z,axis=1)
    hi=mz>0.12
    print(f'animal {a}: frames with median-node z>0.12 m: {hi.sum()} ({hi.mean()*100:.2f}%)')
    if hi.sum():
        idx=np.where(hi)[0]
        # contiguous runs
        br=np.where(np.diff(idx)>1)[0]
        starts=np.r_[idx[0],idx[br+1]]; ends=np.r_[idx[br],idx[-1]]
        L=ends-starts+1
        print('  runs:',len(starts),' longest:',L.max(),' top runs (start,len):',sorted(zip(L,starts))[-5:])
        tl=np.linalg.norm(t[:,a,ix['Neck']]-t[:,a,ix['TTI']],axis=-1)
        print('  neck-TTI in hi frames median %.4f vs all %.4f'%(np.median(tl[hi]),np.median(tl)))
        print('  min node z in hi frames, median: %.4f'%np.median(z[hi].min(1)))
        # jump in centroid at run starts
        c=t[:,a].mean(1)
        for st in starts[:5]:
            if st>0: print('   jump at',st,'dz=%.4f dxy=%.4f'%(c[st,2]-c[st-1,2], np.linalg.norm(c[st,:2]-c[st-1,:2])))
        print('  centroid xy in hi frames (median):',np.median(c[hi,:2],0), ' all:',np.median(c[:,:2],0))
