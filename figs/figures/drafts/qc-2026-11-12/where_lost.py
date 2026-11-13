import json, h5py, numpy as np, sys, glob
s=sys.argv[1]
R=f'/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs/out/tmp/fig8m50_2dc001fe_b3/sync_stale20_dist25/{s}.json'
DET=f'/root/vast/eric/luc3d-bench/outputs/bmimica/det_h5_bvis5/{s}'
GT=f'/root/vast/eric/luc3d-bench/outputs/bmimica/gt/{s}'
o=json.load(open(R)); cams=o['cameras']
det={c:h5py.File(f'{DET}/{c}_predictions.h5','r')['tracks'][0][...] for c in cams}  # F,4,15,2
gt={}
for c in cams:
    with h5py.File(f'{GT}/{c}/proofread.analysis.h5','r') as h: gt[c]=np.transpose(h['tracks'][:],(3,0,2,1)) # F,2,15,2
F=min(min(len(det[c]) for c in cams), min(len(gt[c]) for c in cams), len(o['frames']))
# per frame per cam: list of (lucid_id, gt_id)
pairs=np.full((F,len(cams),2),-1,int)  # for each lucid id -> gt id matched
for fr in o['frames']:
    f=fr['frame']
    if f>=F: continue
    for key,idv in fr['assignments']:
        c,slot=key.split(':'); ci=cams.index(c); slot=int(slot)
        d=det[c][f,slot]; g=gt[c][f]
        dist=[np.nanmean(np.linalg.norm(d-g[k],axis=-1)) for k in range(2)]
        if np.all(np.isnan(dist)): continue
        k=int(np.nanargmin(dist))
        if dist[k]<30 and idv in (0,1): pairs[f,ci,idv]=k
# global mapping
same=((pairs[...,0]==0)|(pairs[...,1]==1)).sum(); swap=((pairs[...,0]==1)|(pairs[...,1]==0)).sum()
m=np.array([0,1]) if same>=swap else np.array([1,0])
valid=pairs>=0
wrong=valid & (pairs!=m[None,None,:])
wf=wrong.sum((1,2)); vf=valid.sum((1,2))
print(s,'F',F,'frac assigned dets wrong vs global map: %.4f'%(wf.sum()/vf.sum()))
# per camera
for ci,c in enumerate(cams): print('   cam',c,'wrong frac %.4f'%(wrong[:,ci].sum()/max(valid[:,ci].sum(),1)))
# runs of wrong frames
bad=wf>vf/2
idx=np.where(bad)[0]
if len(idx):
    br=np.where(np.diff(idx)>1)[0]; st=np.r_[idx[0],idx[br+1]]; en=np.r_[idx[br],idx[-1]]
    L=en-st+1; o_=np.argsort(-L)[:8]
    print('   longest wrong runs (start,len):',[(int(st[i]),int(L[i])) for i in o_])
# relation to high-z frames
f3=glob.glob(f'/root/vast/eric/BMimica/{s}/*_points3d*.h5')[0]
with h5py.File(f3,'r') as h: t=h['tracks'][:F]
hi=(np.median(t[...,2],axis=2)>0.12).any(1)
print('   hi-z frames: %d (%.2f%%); wrong-majority frames: %d; overlap: %d; wrong dets in hi-z frames: %.1f%% of all wrong'%(hi.sum(),hi.mean()*100,bad.sum(),(bad&hi).sum(), 100*wf[hi].sum()/max(wf.sum(),1)))
# GT occupancy/valid gt in hi frames vs others
gtv=np.stack([np.isfinite(gt[c][:F,:,:,0]).any(-1) for c in cams],1)  # F,C,2
print('   GT instances present: hi-z %.3f, other %.3f'%(gtv[hi].mean() if hi.any() else np.nan, gtv[~hi].mean()))
# mapping from first 40k frames
p=pairs[:40000]
same=((p[...,0]==0)|(p[...,1]==1)).sum(); swap=((p[...,0]==1)|(p[...,1]==0)).sum()
m0=np.array([0,1]) if same>=swap else np.array([1,0])
w=valid&(pairs!=m0[None,None,:])
for a,b in [(0,40000),(40000,80000),(80000,105000),(105000,115300),(115300,130000),(130000,155000),(155000,F)]:
    print('   window %6d-%6d wrong-vs-early-map %.3f'%(a,b,w[a:b].sum()/max(valid[a:b].sum(),1)))
