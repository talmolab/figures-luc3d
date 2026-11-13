import h5py, numpy as np, glob, sys
s=sys.argv[1]
cams=["21241563","21369048","21372315","21372316","22085397"]
f3=glob.glob(f'/root/vast/eric/BMimica/{s}/*_points3d*.h5')[0]
with h5py.File(f3,'r') as h: t=h['tracks'][:]
F=len(t)
hiA=np.median(t[...,2],axis=2)>0.12  # F,2
for c in cams:
    with h5py.File(f'/root/vast/eric/luc3d-bench/outputs/bmimica/gt/{s}/{c}/proofread.analysis.h5','r') as h: g=np.transpose(h['tracks'][:],(3,0,2,1))
    with h5py.File(f'/root/vast/eric/luc3d-bench/outputs/bmimica/det_h5_bvis5/{s}/{c}_predictions.h5','r') as h: d=h['tracks'][0][...]
    n=min(len(g),len(d),F)
    g=g[:n]; d=d[:n]
    # for each gt instance: min over detections of median keypoint distance
    dist=np.linalg.norm(g[:,:,None]-d[:,None,:],axis=-1)  # n,2,4,15
    md=np.nanmedian(dist,axis=-1)  # n,2,4
    best=np.nanmin(np.where(np.isnan(md),np.inf,md),axis=-1)  # n,2
    hi=hiA[:n]
    def q(x): x=x[np.isfinite(x)]; return '%.1f/%.1f px (n=%d)'%(np.median(x),np.percentile(x,90),len(x)) if len(x) else 'na'
    print(s,c,' GT->nearest det median/p90: elevated-instance',q(best[hi]),' | normal',q(best[~hi]))
