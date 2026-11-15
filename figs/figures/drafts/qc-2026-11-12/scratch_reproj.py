import h5py, numpy as np, sys
s=sys.argv[1]
f=f'/root/vast/eric/BMimica/scratch/triangulation_results/{s}_triangulated.h5'
with h5py.File(f,'r') as h:
    cams=list(h['reproj'].keys())
    print(s, 'per-camera median reprojection error (px) of the scratch DLT pass, by frame window')
    for c in cams:
        p=h['pts_2d'][c][:]; r=h['reproj'][c][:]
        e=np.linalg.norm(p-r,axis=-1)  # F,2,15
        row=[]
        for a,b in [(0,40000),(40000,60000),(60000,80000),(80000,100000),(100000,115300),(115300,140000),(140000,180000)]:
            x=e[a:b]; x=x[np.isfinite(x)]; row.append('%6.1f'%np.median(x))
        print('  ',c,' '.join(row))
