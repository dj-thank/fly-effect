"""Render a recorded MuJoCo pose; never generate or interpolate movement."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
import mujoco as mj

from organism_core.config import HOME as ROOT, DATA
os.environ.setdefault('FLYGYM_ASSET_CACHE_DIR',str(ROOT/'cache/flygym-assets'))

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--frame',type=int,default=-1)
    args=p.parse_args();path=args.run.resolve()
    from organism_core.body import Body
    report=json.loads((path/'result.json').read_text(encoding='utf-8'))
    body=Body(report['identity']['settings'].get('joint_profile','generic'),report['identity']['settings'].get('muscle_model','antagonist'))
    if report['identity']['body_sha256']!=body.digest:raise ValueError('Recorded body differs from renderer')
    with np.load(path/'observation.npz',allow_pickle=False) as data:
        pose=data['frame_qpos'][args.frame].copy();tick=int(data['frame_ticks'][args.frame])
    body.d.qpos[:]=pose;mj.mj_forward(body.m,body.d)
    camera=mj.MjvCamera();camera.lookat[:]=pose[:3];camera.distance=7;camera.azimuth=115;camera.elevation=-25
    with mj.Renderer(body.m,height=480,width=640) as renderer:
        renderer.update_scene(body.d,camera=camera)
        pixels=renderer.render().copy()
    from PIL import Image
    out=path/f'pose-{tick:06d}.png';Image.fromarray(pixels).save(out)
    (out.with_suffix('.json')).write_text(json.dumps({'source':str(path/'observation.npz'),
        'source_sha256':hashlib.sha256((path/'observation.npz').read_bytes()).hexdigest(),
        'tick':tick,'time_s':tick*.0001,'body_sha256':body.digest,'pose_interpolated':False,
        'walking_claimed':False},indent=2)+'\n',encoding='utf-8')
    print(out)

if __name__=='__main__':main()
