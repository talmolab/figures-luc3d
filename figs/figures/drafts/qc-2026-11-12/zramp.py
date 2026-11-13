import h5py, numpy as np, glob
f=glob.glob('/root/vast/eric/BMimica/20250904_183921/*_points3d*.h5')[0]
with h5py.File(f,'r') as h: t=h['tracks'][:]
c=t[:,0].mean(1)
for st in [44342, 63845]:
    w=range(st-450,st+60,30)
    print(st,' '.join('%d:%.3f'%(i-st,c[i,2]) for i in w))
# when was hi first/last in session
z=np.median(t[...,2],axis=2)
hi=(z>0.12).any(1)
idx=np.where(hi)[0]; print('first',idx.min(),'last',idx.max(),'of',len(t))
# height distribution of hi frames
print(np.percentile(np.median(t[:,0,:,2],1)[np.median(t[:,0,:,2],1)>0.12],[5,50,95]))
# other animal's z when animal0 is high
a0=np.median(t[:,0,:,2],1)>0.12
print('animal1 median z when a0 high:',np.median(np.median(t[a0,1,:,2],1)))
d=np.linalg.norm(t[:,0].mean(1)-t[:,1].mean(1),axis=-1)
print('inter-animal dist when a0 high median',np.median(d[a0]))
