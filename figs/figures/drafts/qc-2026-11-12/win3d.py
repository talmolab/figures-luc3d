import h5py, numpy as np, glob, sys
s=sys.argv[1]
f3=glob.glob(f'/root/vast/eric/BMimica/{s}/*_points3d*.h5')[0]
with h5py.File(f3,'r') as h: t=h['tracks'][:]; nodes=[n.decode() for n in h['node_names'][:]]
ix={n:i for i,n in enumerate(nodes)}
for a,b in [(0,40000),(40000,60000),(60000,80000),(80000,100000),(100000,115300),(115300,140000),(140000,180000)]:
    w=t[a:b]; tti=w[:,:,ix['TTI']]
    z=tti[...,2].ravel(); x=tti[...,0].ravel(); y=tti[...,1].ravel()
    body=np.linalg.norm(w[:,:,ix['Neck']]-w[:,:,ix['TTI']],axis=-1).ravel()
    ears=np.linalg.norm(w[:,:,ix['Ear_L']]-w[:,:,ix['Ear_R']],axis=-1).ravel()
    print(f'{a:6d}-{b:6d} TTI z p1/p50/p99 {np.percentile(z,1)*1000:5.1f}/{np.percentile(z,50)*1000:5.1f}/{np.percentile(z,99)*1000:6.1f} mm  x[{np.percentile(x,0.5)*1000:5.0f},{np.percentile(x,99.5)*1000:4.0f}] y[{np.percentile(y,0.5)*1000:5.0f},{np.percentile(y,99.5)*1000:4.0f}] body {np.median(body)*1000:.1f} ears {np.median(ears)*1000:.1f} mm')
