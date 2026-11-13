import h5py, numpy as np, glob, sys
s=sys.argv[1]
cams=["21241563","21369048","21372315","21372316","22085397"]
maps=[]
for c in cams:
    with h5py.File(f'/root/vast/eric/luc3d-bench/outputs/bmimica/gt/{s}/{c}/proofread.analysis.h5','r') as h: g=np.transpose(h['tracks'][:],(3,0,2,1))
    a=glob.glob(f'/root/vast/eric/BMimica/{s}/{c}/*.analysis.h5')[0]
    with h5py.File(a,'r') as h: r=np.transpose(h['tracks'][:],(3,0,2,1))
    n=min(len(g),len(r)); g=g[:n]; r=r[:n,:2]   # only first 2 slots
    with np.errstate(all='ignore'):
        d00=np.nanmedian(np.linalg.norm(r[:,0]-g[:,0],axis=-1),-1); d01=np.nanmedian(np.linalg.norm(r[:,0]-g[:,1],axis=-1),-1)
    m=np.where(np.isnan(d00)|np.isnan(d01),-1,(d01<d00).astype(int))  # 1 = raw slot0 is GT animal 1
    maps.append(m)
n=min(len(m) for m in maps); M=np.stack([m[:n] for m in maps],1)
ok=(M>=0).all(1)
agree=(M[ok]==M[ok][:,[0]]).all(1)
print(s,'frames where all 5 cams raw SLEAP slot0 maps to the same proofread animal: %.1f%%'%(100*agree.mean()),
      ' per-cam share slot0->animal1:',' '.join('%.2f'%(M[ok][:,i].mean()) for i in range(5)))
