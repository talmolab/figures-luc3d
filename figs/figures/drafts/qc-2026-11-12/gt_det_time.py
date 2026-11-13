import h5py, numpy as np, glob, sys, json
s=sys.argv[1]
cams=["21241563","21369048","21372315","21372316","22085397"]
f3=glob.glob(f'/root/vast/eric/BMimica/{s}/*_points3d*.h5')[0]
with h5py.File(f3,'r') as h: t=h['tracks'][:]
F=len(t); bad=[]
for c in cams:
    with h5py.File(f'/root/vast/eric/luc3d-bench/outputs/bmimica/gt/{s}/{c}/proofread.analysis.h5','r') as h: g=np.transpose(h['tracks'][:],(3,0,2,1))
    with h5py.File(f'/root/vast/eric/luc3d-bench/outputs/bmimica/det_h5_bvis5/{s}/{c}_predictions.h5','r') as h: d=h['tracks'][0][...]
    n=min(len(g),len(d),F); F=n
    dist=np.linalg.norm(g[:n,:,None]-d[:n,None,:],axis=-1)
    with np.errstate(all='ignore'):
        md=np.nanmedian(dist,axis=-1); best=np.nanmin(np.where(np.isnan(md),np.inf,md),axis=-1)
    bad.append(best>20)  # n,2
bad=np.stack([b[:F] for b in bad],1)  # F,C,2
inst_bad=bad.sum(1)>=3   # instance off in >=3 of 5 cams
print(s,'instance-frames where reprojected proofread 3D is >20px from every detection in >=3 cams: %.2f%%'%(100*inst_bad.mean()))
fb=inst_bad.any(1)
for a,b in [(0,40000),(40000,80000),(80000,105000),(105000,115300),(115300,155000),(155000,F)]:
    print('   frames %6d-%6d: %.2f%%'%(a,b,100*fb[a:b].mean()))
json.dump({'inst_bad_frac':float(inst_bad.mean())},open(f'/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/0966bec4-7dd1-4a61-b768-aef43cefa6e8/scratchpad/qc/gtdet_{s}.json','w'))
