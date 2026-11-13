import h5py, numpy as np, glob, os, sys, json
root='/root/vast/eric/BMimica'
out={}
for sd in sorted(glob.glob(root+'/2025*')):
    s=os.path.basename(sd)
    f=glob.glob(sd+'/*_points3d_translated_rotated_metric.h5')
    if not f: print(s,'no3d'); continue
    with h5py.File(f[0],'r') as h:
        t=h['tracks'][::5]  # F,2,15,3
        nodes=[n.decode() if isinstance(n,bytes) else n for n in h['node_names'][:]]
        fps=float(np.array(h['recording_frame_rate'])) if 'recording_frame_rate' in h else None
        nf=h['tracks'].shape[0]
    ix={n:i for i,n in enumerate(nodes)}
    z=t[...,2]; zf=z[np.isfinite(z)]
    def seg(a,b):
        d=np.linalg.norm(t[:,:,ix[a]]-t[:,:,ix[b]],axis=-1); d=d[np.isfinite(d)]; return float(np.median(d)) if d.size else np.nan
    xy=t[...,:2].reshape(-1,2); xy=xy[np.isfinite(xy).all(1)]
    r=dict(nf=nf,fps=fps,zp50=float(np.percentile(zf,50)),zp99=float(np.percentile(zf,99)),zp1=float(np.percentile(zf,1)),
      xrange=float(np.percentile(xy[:,0],99.5)-np.percentile(xy[:,0],0.5)),yrange=float(np.percentile(xy[:,1],99.5)-np.percentile(xy[:,1],0.5)),
      ears=seg('Ear_L','Ear_R'),nose_head=seg('Nose','Head'),body=seg('Neck','TTI'),trunk_tti=seg('Trunk','TTI'),
      nan=float(np.isnan(t).mean()),
      tailtip_zp99=float(np.nanpercentile(t[:,:,ix['TailTip'],2],99)), neck_zp99=float(np.nanpercentile(t[:,:,ix['Neck'],2],99)),
      nose_zp99=float(np.nanpercentile(t[:,:,ix['Nose'],2],99)))
    out[s]=r
    print(s, ' '.join(f'{k}={v:.4f}' if isinstance(v,float) else f'{k}={v}' for k,v in r.items()), flush=True)
json.dump(out,open(sys.argv[1],'w'),indent=1)
