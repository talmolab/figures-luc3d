import h5py, numpy as np, glob, sys
s=sys.argv[1]; frames=[int(x) for x in sys.argv[2:]]
f3=glob.glob(f'/root/vast/eric/BMimica/{s}/*_points3d*.h5')[0]
with h5py.File(f3,'r') as h: t=h['tracks'][:]
c=np.nanmedian(t,axis=2)  # F,2,3
d_same=np.linalg.norm(c[1:]-c[:-1],axis=-1).max(1)
d_swap=np.maximum(np.linalg.norm(c[1:,0]-c[:-1,1],axis=-1),np.linalg.norm(c[1:,1]-c[:-1,0],axis=-1))
# frames where swap explanation is far better than same: identity swap in 3D
sw=np.where((d_swap<0.5*d_same)&(d_same>0.02))[0]+1
print(s,'3D track discontinuities consistent with an identity swap:',len(sw), sw[:30].tolist())
big=np.where(d_same>0.03)[0]+1
print('  frames with >3cm centroid jump:',len(big), big[:30].tolist())
for f in frames:
    print('  frame',f,'same-jump %.4f swap-jump %.4f  inter-animal dist %.4f'%(d_same[f-1],d_swap[f-1],np.linalg.norm(c[f,0]-c[f,1])))
