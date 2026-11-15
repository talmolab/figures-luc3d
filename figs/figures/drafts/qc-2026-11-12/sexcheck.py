import h5py, numpy as np, glob, sys
s=sys.argv[1]; cut=[int(x) for x in sys.argv[2:]]
f3=glob.glob(f'/root/vast/eric/BMimica/{s}/*_points3d*.h5')[0]
with h5py.File(f3,'r') as h: t=h['tracks'][:]; nodes=[n.decode() for n in h['node_names'][:]]; tn=[x.decode() for x in h['track_names'][:]]
ix={n:i for i,n in enumerate(nodes)}
bl=np.linalg.norm(t[:,:,ix['Nose']]-t[:,:,ix['TTI']],axis=-1)*1000
ed=np.linalg.norm(t[:,:,ix['Ear_L']]-t[:,:,ix['Ear_R']],axis=-1)*1000
edges=[0]+cut+[len(t)]
print(s,'tracks',tn)
for a,b in zip(edges[:-1],edges[1:]):
    print(f'  {a:6d}-{b:6d} nose-TTI p75 mm: track0 {np.percentile(bl[a:b,0],75):.1f} track1 {np.percentile(bl[a:b,1],75):.1f} | ear dist med: {np.median(ed[a:b,0]):.1f} {np.median(ed[a:b,1]):.1f}')
