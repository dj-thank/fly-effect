"""Encode recorded poses at 30 fps; no gait interpolation or simulated motion added."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
import numpy as np
import mujoco as mj
from organism_core.body import Body

from organism_core.config import HOME as ROOT, DATA

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();directory=args.run.resolve()
    report=json.loads((directory/'result.json').read_text(encoding='utf-8'));settings=report['identity']['settings']
    body=Body(settings['joint_profile'],settings.get('muscle_model','antagonist'))
    if body.digest!=report['identity']['body_sha256']:raise ValueError('Recorded body differs')
    with np.load(directory/'observation.npz',allow_pickle=False) as a:
        q=a['frame_qpos'].copy();ticks=a['frame_ticks'].copy()
    if hashlib.sha256(q.tobytes()).hexdigest()!=report['observation_hashes']['frame_qpos']:raise ValueError('Changed poses')
    frames=round(report['ticks']*.0001*30);selected=np.rint(np.arange(frames)*1000/30).astype(int)
    if selected.max()>=len(q):raise ValueError('Insufficient recorded poses')
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise ValueError('FFmpeg required')
    out=directory/'actual-motion.mp4'
    if out.exists():raise ValueError('Output already exists')
    command=[ffmpeg,'-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','640x480','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    camera=mj.MjvCamera();camera.lookat[:]=q[0,:3];camera.distance=7;camera.azimuth=115;camera.elevation=-25
    with mj.Renderer(body.m,height=480,width=640) as renderer:
        for index in selected:
            body.d.qpos[:]=q[index];mj.mj_forward(body.m,body.d)
            renderer.update_scene(body.d,camera=camera);process.stdin.write(renderer.render().tobytes())
    process.stdin.close();error=process.stderr.read().decode('utf-8',errors='replace');code=process.wait()
    if code:raise RuntimeError(error)
    receipt={'source':str(directory/'observation.npz'),'source_sha256':hashlib.sha256((directory/'observation.npz').read_bytes()).hexdigest(),
        'selected_source_ticks':ticks[selected].tolist(),'fps':30,'frames':frames,'pose_interpolation':False,'camera':'fixed',
        'video_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'walking_claimed':False}
    out.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(out)

if __name__=='__main__':main()
