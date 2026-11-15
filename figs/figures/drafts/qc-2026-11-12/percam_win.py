import h5py, numpy as np, sys
s=sys.argv[1]
cams=["21241563","21369048","21372315","21372316","22085397"]
wins=[(0,40000),(40000,60000),(60000,80000),(80000,100000),(100000,115300),(115300,140000),(140000,180000)]
print(s, 'median px GT(reprojected proofread 3D)->nearest detection, per camera per window')
for c in cams:
    with h5py.File(f'/root/vast/eric/luc3d-bench/outputs/bmimica/gt/{s}/{c}/proofread.analysis.h5','r') as h: g=np.transpose(h['tracks'][:],(3,0,2,1))
    with h5py.File(f'/root/vast/eric/luc3d-bench/outputs/bmimica/det_h5_bvis5/{s}/{c}_predictions.h5','r') as h: d=h['tracks'][0][...]
    row=[]
    for a,b in wins:
        fr=np.arange(a,min(b,len(g),len(d)),20)
        dist=np.linalg.norm(g[fr][:,:,None]-d[fr][:,None,:],axis=-1)
        with np.errstate(all='ignore'):
            md=np.nanmedian(dist,axis=-1); bb=np.nanmin(np.where(np.isnan(md),np.inf,md),axis=-1)
        bb=bb[np.isfinite(bb)]; row.append('%6.1f'%np.median(bb))
    print('  ',c,' '.join(row))
