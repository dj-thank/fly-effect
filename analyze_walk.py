"""Report locomotion diagnostics; single development trials never certify F06."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

def sustained_onsets(contacts,min_frames=5):
    counts=np.zeros(6,dtype=int)
    for leg in range(6):
        off_run=min_frames;on_run=0;armed=True
        for value in contacts[:,leg]:
            if value:
                on_run+=1
                if on_run>=min_frames and armed:
                    counts[leg]+=1;armed=False
                off_run=0
            else:
                off_run+=1;on_run=0
                if off_run>=min_frames:armed=True
    return counts

def main():
    import mujoco as mj
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    from organism_core.body import Body
    report=json.loads((args.run/'result.json').read_text(encoding='utf-8'))
    body=Body(report['identity']['settings'].get('joint_profile','generic'),report['identity']['settings'].get('muscle_model','antagonist'));mj.mj_forward(body.m,body.d)
    if body.digest!=report['identity']['body_sha256']:raise ValueError('Diagnostic model differs from recorded body')
    points=[]
    for i in range(body.m.ngeom):
        name=mj.mj_id2name(body.m,mj.mjtObj.mjOBJ_BODY,int(body.m.geom_bodyid[i])) or ''
        if not any(part in name for part in ('/c_head','/c_thorax','/c_abdomen')) or body.m.geom_type[i]!=mj.mjtGeom.mjGEOM_MESH:continue
        mesh=int(body.m.geom_dataid[i]);start=body.m.mesh_vertadr[mesh];n=body.m.mesh_vertnum[mesh]
        vertices=body.m.mesh_vert[start:start+n]
        points.append(vertices@body.d.geom_xmat[i].reshape(3,3).T+body.d.geom_xpos[i])
    body_length=float(np.ptp(np.concatenate(points)[:,0]))
    if not 0<body_length<20:raise ValueError('Invalid body length')
    with np.load(args.run/'observation.npz',allow_pickle=False) as archive:
        q=archive['frame_qpos'];contact=archive['frame_contacts'];ticks=archive['frame_ticks']
        if not np.all(np.diff(ticks)==10):raise ValueError('Expected 1ms trajectory frames')
        net=float(np.linalg.norm(q[-1,:2]-body.d.qpos[:2]))
        raw=np.sum(contact[1:] & ~contact[:-1],axis=0)
        filtered=sustained_onsets(contact)
        force=float(archive['frame_muscle_force'].max())
        up_z=1-2*(q[:,4]**2+q[:,5]**2)
        evaluation=ticks>=2000
        upright_fraction=float(np.mean(up_z[evaluation]>=.5)) if evaluation.any() else 0.
    nonfoot_frames=0;nonfoot_geoms={};evaluated_frames=0
    for pose,tick in zip(q,ticks):
        if tick<2000:continue
        evaluated_frames+=1;body.d.qpos[:]=pose;mj.mj_fwdPosition(body.m,body.d)
        hit=set()
        for i in range(body.d.ncon):
            contact=body.d.contact[i]
            if contact.dist>0:continue
            names=[mj.mj_id2name(body.m,mj.mjtObj.mjOBJ_GEOM,int(g)) or '' for g in (contact.geom1,contact.geom2)]
            if 'ground_plane' not in names:continue
            name=names[1] if names[0]=='ground_plane' else names[0]
            if not any(f'/{leg}_tarsus' in name for leg in ('lf','lm','lh','rf','rm','rh')):hit.add(name)
        nonfoot_frames+=bool(hit)
        for name in hit:nonfoot_geoms[name]=nonfoot_geoms.get(name,0)+1
    nonfoot_fraction=nonfoot_frames/evaluated_frames if evaluated_frames else 1.
    checks={'duration_at_least_3s':report['ticks']>=30000,'three_body_lengths':net>=3*body_length,
            'three_sustained_contacts_each_leg':bool((filtered>=3).all()),
            'required_motor_mapping_complete':report['motor_coverage']['unresolved_required']==0,
            'no_external_perturbation':report['identity']['settings']['perturbation'] is None,
            'upright_not_tumbling':upright_fraction>=.9,'nonfoot_ground_contact_at_most_10pct':nonfoot_fraction<=.1,
            'actual_motor_and_muscle_activity':report['motor_spikes']>0 and force>0}
    result={'body_length_mm_from_core_mesh':body_length,'net_distance_mm':net,'body_lengths':net/body_length,
            'raw_contact_onsets':raw.tolist(),'sustained_contact_onsets':filtered.tolist(),
            'upright_fraction_after_settle':upright_fraction,'nonfoot_contact_fraction_after_settle':nonfoot_fraction,
            'nonfoot_contact_frame_counts':nonfoot_geoms,'nonfoot_contact_scope':'Geometric contact at recorded qpos, not reconstructed contact force; excludes first200ms',
            'upright_definition':'Dorsal axis within 60deg of vertical for >=90% of frames after initial200ms; flat-ground diagnostic',
            'contact_filter':'5 consecutive 1ms contact frames and >=5ms release; diagnostic anti-chatter filter, not a gait controller',
            'checks':checks,'single_trial_criteria_met':all(checks.values()),'walking_passed':False,
            'reason':'Development seed; formal F06 additionally needs full output mapping and 4 of 5 acceptance seeds',
            'source_sha256':hashlib.sha256((args.run/'observation.npz').read_bytes()).hexdigest()}
    (args.run/'walking-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
