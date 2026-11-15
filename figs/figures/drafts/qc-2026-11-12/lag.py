import h5py, numpy as np
s='20250904_183921'; c='21241563'
with h5py.File(f'/root/vast/eric/luc3d-bench/outputs/bmimica/gt/{s}/{c}/proofread.analysis.h5','r') as h: g=np.transpose(h['tracks'][:],(3,0,2,1))
with h5py.File(f'/root/vast/eric/luc3d-bench/outputs/bmimica/det_h5_bvis5/{s}/{c}_predictions.h5','r') as h: d=h['tracks'][0][...]
def score(fr, lag):
    ff=fr+lag; ok=(ff>=0)&(ff<len(d))
    gg=g[fr[ok]]; dd=d[ff[ok]]
    dist=np.linalg.norm(gg[:,:,None]-dd[:,None,:],axis=-1)
    with np.errstate(all='ignore'):
        md=np.nanmedian(dist,axis=-1); b=np.nanmin(np.where(np.isnan(md),np.inf,md),axis=-1)
    b=b[np.isfinite(b)]; return np.median(b), (b<20).mean()
for lo,hi in [(10000,20000),(45000,55000),(60000,70000),(85000,95000),(108000,112000),(130000,140000)]:
    fr=np.arange(lo,hi,50)
    res=[(lag,)+score(fr,lag) for lag in [-3000,-1500,-600,-300,-150,-60,-30,-10,-3,0,3,10,30,60,150,300,600,1500,3000]]
    best=min(res,key=lambda r:r[1])
    print(f'{lo}-{hi}: lag0 med={res[9][1]:.1f}px ok={res[9][2]:.2f}; best lag {best[0]} med={best[1]:.1f}px ok={best[2]:.2f}')
