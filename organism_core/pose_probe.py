"""Bounded diagnostic initial-pose search; not a motion policy or neural fit.

The fixed 3x3 hip/knee grid changes only initial coordinates within EXISTING
joint limits. Vertical placement brackets first active ground contact. No mass,
force limit, contact margin, neural weight, or accepted gait criterion is fitted.
"""
from __future__ import annotations
import time
import numpy as np
import mujoco as mj
from .contacts import is_active_contact
from .force_audit import audit_static_forces
from .support_diagnostics import contact_columns, evaluate_support, LEGS


def ground_contacts(model, data):
    """Per-geom forces on the body after forward dynamics; not just membership."""
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if not is_active_contact(contact):
            continue
        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(g)) or ''
                 for g in (contact.geom1, contact.geom2)]
        if 'ground_plane' not in names:
            continue
        first = names[0] == 'ground_plane'
        name = names[1 if first else 0]
        force = np.zeros(6)
        mj.mj_contactForce(model, data, index, force)
        world = (1 if first else -1)*contact.frame.reshape(3,3).T@force[:3]
        rows.append({'geom': name, 'foot': any(f'/{leg}_tarsus' in name for leg in LEGS),
                     'distance_native': float(contact.dist),
                     'includemargin_native': float(contact.includemargin),
                     'world_force_native': world.tolist(), 'normal_force_native': float(force[0])})
    return rows


def first_contact_height(model, data, qpos, *, deadline):
    """Fixed-orientation vertical bracket, for one free-root body over a plane.

Private scratch data may be changed; model and qpos input are not changed.
Only contact membership is queried, so no dynamics force is claimed here.
"""
    if (model.njnt < 1 or model.jnt_type[0] != mj.mjtJoint.mjJNT_FREE
            or model.jnt_qposadr[0] != 0):
        raise ValueError('Expected one root free joint at qpos[0:7]')
    q = np.asarray(qpos, dtype=float).copy()
    if q.shape != (model.nq,) or not np.isfinite(q).all():
        raise ValueError('Finite matching position vector required')
    def active(z):
        if time.monotonic() > deadline:
            raise TimeoutError('Pose-probe wall budget exceeded')
        data.qpos[:] = q; data.qpos[2] = z
        mj.mj_fwdPosition(model, data)
        return any(is_active_contact(c) and 'ground_plane' in [
            mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)]
            for c in data.contact[:data.ncon])
    lo, hi = float(q[2]-2.), float(q[2]+2.)
    if active(hi) or not active(lo):
        raise ValueError('First ground contact not bracketed within +/-2 native length units')
    for _ in range(25):
        mid = (lo+hi)/2
        if active(mid): lo = mid
        else: hi = mid
    return lo


def scan_initial_poses(body, transmission, q0, fmax, l0, *, deadline):
    """Nine preregistered poses and support evaluations; source state unchanged.

All coordinates are diagnostic placements, NOT achieved movements. Root force
variables are NOT added to the support solver. A feasible static pose is still
not stable dynamic support, autonomous walking, or a biological calibration.
"""
    from .body import Body
    m = body.m
    probe = Body.__new__(Body); probe.m = m; probe.d = mj.MjData(m)
    d = probe.d
    arrays, rows = {}, []
    for hip in (-.3, 0., .3):
        for knee in (-.3, 0., .3):
            if time.monotonic() > deadline: raise TimeoutError('Pose-probe wall budget exceeded')
            q = np.asarray(q0).copy()
            row = {'index': len(rows), 'hip_offset_rad': hip, 'knee_offset_rad': knee,
                   'artificial_initial_placement': True, 'walking_claimed': False}
            clipped = 0
            for leg in LEGS:
                for kind, offset in (('hip',hip),('knee',knee)):
                    g = body.joint_geometry[leg+':'+kind]
                    lower, upper = m.jnt_range[g['joint_id']]
                    desired = q[g['qpos']]+offset*g['extension_sign']
                    q[g['qpos']] = np.clip(desired,lower+1e-4,upper-1e-4)
                    clipped += int(q[g['qpos']] != desired)
            row['clipped_joint_count'] = clipped
            try:
                height = first_contact_height(m,d,q,deadline=deadline)
            except ValueError as exc:
                row.update(status='not_bracketed',reason=str(exc)); rows.append(row); continue
            samples, eligible = [], []
            for depth in (.001,.005,.01,.02,.04):
                d.qpos[:] = q; d.qpos[2] = height-depth; d.qvel[:] = 0.
                d.qfrc_applied[:] = 0.; d.xfrc_applied[:] = 0.
                mj.mj_forward(m,d)
                _,_,contacts,nonfoot = contact_columns(m,d)
                legs = [leg for leg in LEGS if any(f'/{leg}_tarsus' in c['geom'] for c in contacts)]
                samples.append({'depth_native':depth,'foot_legs':legs,'nonfoot_contacts':nonfoot})
                if contacts and not nonfoot:
                    eligible.append((len(legs),depth,d.qpos.copy()))
            row.update(first_contact_height_native=height,samples=samples)
            if not eligible:
                row['status']='no_foot_only_pose_in_sampled_depths'; rows.append(row); continue
            # Prefer more feet, then the shallowest tested placement; never optimize walking.
            _,depth,chosen = sorted(eligible,key=lambda x:(-x[0],x[1]))[0]
            d.qpos[:] = chosen; d.qvel[:] = 0.; mj.mj_forward(m,d)
            limits = fmax*np.exp(-((d.ten_length/l0-1)/.5)**2)
            report, inputs = evaluate_support(probe,transmission,limits)
            prefix = f'pose_{row["index"]:02d}_'
            arrays.update({prefix+k:v for k,v in inputs.items()})
            budget, forces = audit_static_forces(m, chosen)
            if not budget['passed']: raise ValueError('Pose force accounting failed')
            if not np.allclose(forces['support_target'], inputs['support_target'], rtol=1e-12, atol=1e-12):
                raise ValueError('Pose support target differs from force accounting')
            row['force_accounting'] = budget
            arrays.update({prefix+'force_'+k:v for k,v in forces.items()})
            row.update(status=report['status'],selected_depth_native=depth,support=report,
                       active_ground_loads=ground_contacts(m,d))
            rows.append(row)
    result = {'kind':'diagnostic_initial_pose_grid','hip_offsets_rad':[-.3,0.,.3],
              'knee_offsets_rad':[-.3,0.,.3], 'pose_count':len(rows), 'poses':rows,
              'feasible_pose_count':sum(r['status']=='feasible_at_tested_pose' for r in rows),
              'foot_only_pose_count':sum('support' in r for r in rows),
              'biological_validation':False,'walking_claimed':False,
              'scope':'Artificial initial placements; same model and fixed force bounds; not a movement controller.'}
    return result, arrays


def run_pose_probe(out, study, *, wall_limit=120.):
    """Reconstruct the recorded static pose, then run the fixed initial-pose grid.

The input observation hash and exact body digest are required. Reported loads
are recomputed at zero velocity/zero applied force, not replayed historic forces.
"""
    import hashlib
    import json
    from pathlib import Path
    from .body import Body
    from .transmission import SiteTendonTransmission
    from .calibration import write_result
    if not np.isfinite(wall_limit) or not 0 < wall_limit <= 300:
        raise ValueError('Wall limit must be finite in (0,300] seconds')
    out, study = Path(out), Path(study)
    out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic(); deadline = started+wall_limit
    result = {'schema':1,'kind':'diagnostic_support_pose_probe','status':'running',
              'biological_validation':False,'walking_claimed':False,'CNS_executed':False,
              'wall_limit_s':wall_limit,'functional_gates_automatically_passed':[]}
    arrays = {}
    try:
        raw = (study/'result.json').read_bytes(); source = json.loads(raw)
        if source['status'] != 'completed':raise ValueError('Completed source study required')
        artifact = study/'observations.npz'
        with artifact.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest != source['observations']['sha256']:raise ValueError('Source observation hash mismatch')
        result['source_result_sha256']=hashlib.sha256(raw).hexdigest()
        result['source_observations_sha256']=digest
        body=Body('muscle_compliance','tendon_candidate');m,d=body.m,body.d
        if body.digest != source['body_sha256']:raise ValueError('Recorded body differs from probe model')
        result['body_sha256']=body.digest
        with np.load(artifact,allow_pickle=False) as archive:
            q0=archive['neutral_qpos'].copy(); recorded=archive['support_qpos'].copy()
            target=archive['support_target'].copy()
        if any(q.shape!=(m.nq,) or not np.isfinite(q).all() for q in (q0,recorded)):
            raise ValueError('Finite matching source poses required')
        d.qpos[:]=q0;d.qvel[:]=0.;mj.mj_forward(m,d);l0=d.ten_length.copy()
        trans=SiteTendonTransmission(m);fmax=np.asarray(source['maximum_tensions_native'],dtype=float)
        if fmax.shape!=(m.ntendon,) or not np.isfinite(fmax).all() or np.any(fmax<=0):
            raise ValueError('Positive matching source tension limits required')
        d.qpos[:]=recorded;d.qvel[:]=0.;d.qfrc_applied[:]=0.;mj.mj_forward(m,d)
        if not np.allclose(d.qfrc_bias-d.qfrc_passive,target,rtol=1e-12,atol=1e-12):
            raise ValueError('Recorded support target does not reproduce')
        result['recorded_pose_loads']=ground_contacts(m,d)
        result['recorded_target_reproduced']=True
        before=body.state()
        budget,forces=audit_static_forces(m,recorded)
        result['recorded_force_accounting']=budget
        arrays.update({'recorded_force_'+k:v for k,v in forces.items()})
        if not budget['passed']:raise ValueError('Recorded force accounting failed')
        if not np.allclose(forces['support_target'],target,rtol=1e-12,atol=1e-12):
            raise ValueError('Recorded target differs from force accounting')
        scan,pose_arrays=scan_initial_poses(body,trans,q0,fmax,l0,deadline=deadline)
        arrays.update(pose_arrays)
        if not np.array_equal(before,body.state()):raise ValueError('Pose scan mutated source state')
        result['source_state_unchanged']=True;result['scan']=scan
        result['code_sha256']=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes()
            for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()
        if time.monotonic()>deadline:raise TimeoutError('Pose scan exceeded wall budget')
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        if arrays:
            path=out/'observations.npz';np.savez_compressed(path,**arrays)
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            result['observations']={'file':path.name,'sha256':digest}
        result['wall_seconds']=time.monotonic()-started
        write_result(out,result)
    return result
