"""Explicitly artificial mechanical experiments on the 90-tendon candidate.

No graph, motor annotations, neural fitting, or acceptance seeds are read.
Static feasibility and prescribed-tension trials do not certify autonomous gait.
"""
from __future__ import annotations
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import time
import numpy as np
import mujoco as mj
from .transmission import SiteTendonTransmission, audit_transmission
from .support_diagnostics import contact_columns, evaluate_support, CONTACT_SEMANTICS
from .calibration import write_result


def _trial(body, transmission, initial, fmax, l0, duration, condition, fixed, deadline):
    """Direct bounded tensions are diagnostic forces, NOT neural muscle commands."""
    m, d = body.m, body.d
    body.restore(initial)
    start = float(d.time); ticks = round(duration/m.opt.timestep)
    rows = []; base_residual = 0.; peak_fraction = 0.
    names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_TENDON, i) for i in range(m.ntendon)]
    weight = float(m.body_mass.sum()*abs(m.opt.gravity[2]))
    def record():
        _, _, contacts, nonfoot = contact_columns(m, d)
        feet, reaction = body.contacts()
        rows.append((float(d.time-start), d.qpos.copy(), d.qvel.copy(),
                     float(d.subtree_com[1,2]), [feet[k] for k in ('lf','lm','lh','rf','rm','rh')],
                     bool(nonfoot), reaction.copy()))
    mj.mj_forward(m, d); record()
    for tick in range(ticks):
        if time.monotonic() > deadline:
            raise TimeoutError('Tendon study wall budget exceeded')
        mj.mj_fwdPosition(m, d)
        length_factor = np.exp(-((d.ten_length/l0-1)/.5)**2)
        capacity = fmax*length_factor
        tensions = np.zeros(m.ntendon)
        t = tick*m.opt.timestep
        envelope = max(0., min(1., t/.02, (.15-t)/.02))
        if condition in ('extensor_pulse', 'flexor_pulse'):
            label = 'LFTibia_extensor' if condition == 'extensor_pulse' else 'LFTibia_flex'
            selected = np.array([label in name for name in names])
            tensions[selected] = .05*envelope*capacity[selected]
        elif condition == 'tonic_2pct':
            tensions = .02*min(1., t/.02)*capacity
        elif condition == 'static_solution':
            tensions = np.minimum(fixed, capacity)
        elif condition != 'passive':
            raise ValueError('Unknown diagnostic condition')
        force = transmission.force(d, tensions)
        base_residual = max(base_residual, float(np.abs(force[:6]).max()))
        if base_residual > 1e-10:
            raise ValueError('Internal tendons produced external free-base force')
        d.qfrc_applied[:] = force
        mj.mj_step(m, d)
        if np.any(d.warning.number) or not np.isfinite(np.r_[d.qpos,d.qvel,d.qacc]).all():
            raise ValueError('Nonfinite physics or solver warning')
        if abs(d.time-(start+(tick+1)*m.opt.timestep)) > 1e-9:
            raise ValueError('Physics clock mismatch')
        peak_fraction = max(peak_fraction, float(np.max(tensions/fmax)))
        if (tick+1)%10 == 0:
            mj.mj_forward(m, d); record()
    arrays = {'time_s': np.array([r[0] for r in rows]), 'qpos': np.array([r[1] for r in rows]),
              'qvel': np.array([r[2] for r in rows]), 'com_height': np.array([r[3] for r in rows]),
              'foot_contacts': np.array([r[4] for r in rows]),
              'nonfoot_contact': np.array([r[5] for r in rows]),
              'ground_reaction': np.array([r[6] for r in rows])}
    q = arrays['qpos']; upright = 1-2*(q[:,4]**2+q[:,5]**2)
    summary = {'ticks': ticks, 'duration_s': duration, 'condition': condition,
               'artificial_prescribed_tensions': condition != 'passive',
               'peak_fraction_of_source_fmax': peak_fraction,
               'maximum_internal_base_residual': base_residual,
               'net_xy_native': float(np.linalg.norm(q[-1,:2]-q[0,:2])),
               'initial_com_height_native': float(arrays['com_height'][0]),
               'final_com_height_native': float(arrays['com_height'][-1]),
               'upright_fraction': float(np.mean(upright>=.5)),
               'nonfoot_contact_fraction': float(np.mean(arrays['nonfoot_contact'])),
               'mean_vertical_support_over_weight': float(np.mean(arrays['ground_reaction'][:,2])/weight),
               'solver_warnings': int(d.warning.number.sum()), 'walking_claimed': False}
    return summary, arrays


def run_study(out, *, duration=.25, wall_limit=240.):
    if not np.isfinite([duration, wall_limit]).all() or not .15 <= duration <= .5 or not 0 < wall_limit <= 600:
        raise ValueError('Duration 0.15--0.5 s and wall budget (0,600] s required')
    if abs(duration-round(duration/.0001)*.0001)>1e-12:
        raise ValueError('Integer 100-us duration required')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started+wall_limit
    result = {'schema': 1, 'kind': '90_tendon_mechanical_study', 'status': 'running',
              'biological_validation': False, 'walking_claimed': False,
              'CNS_executed': False, 'functional_gates_automatically_passed': [],
              'duration_s': duration, 'wall_limit_s': wall_limit,
              'contact_semantics': CONTACT_SEMANTICS,
              'assumptions': ['LF muscle geometry transferred to all six legs.',
                'Source maximum forces are transfer priors, not fitted biological measurements.',
                'Dynamic trials prescribe tensions directly; neural and activation dynamics are bypassed.',
                'Length envelope uses neutral tendon length surrogate and width 0.5; no velocity law.',
                'Static support uses only foot contacts, excludes joint friction/limit reaction variables.']}
    arrays = {}
    try:
        from .body import Body, UNIT_CONTRACT
        from .config import HOME
        body = Body('muscle_compliance','tendon_candidate'); m,d = body.m,body.d
        if m.ntendon != 90 or m.nu != 0 or abs(m.opt.timestep-.0001)>1e-12:
            raise ValueError('Expected unactuated 90-tendon, 100-us candidate')
        trans = SiteTendonTransmission(m); mj.mj_forward(m,d)
        original = body.state(); q0 = d.qpos.copy(); l0 = d.ten_length.copy()
        audit = json.loads((HOME/'runs/source-tendon-audit/results.json').read_text())
        gains = {r['muscle']: r['source_gainprm'][2] for r in audit['muscles']}
        names = [mj.mj_id2name(m,mj.mjtObj.mjOBJ_TENDON,i) for i in range(m.ntendon)]
        fmax = np.array([gains[n.split('_',2)[2].removesuffix('_tendon')] for n in names])
        if not np.isfinite(fmax).all() or np.any(fmax<=0):
            raise ValueError('Invalid source force limits')
        result.update(body_sha256=body.digest, units=UNIT_CONTRACT,
                      source_hashes=body.model_source_hashes, tendon_names=names,
                      maximum_tensions_native=fmax.tolist(),
                      versions={n:metadata.version(n) for n in ('numpy','mujoco','flygym','scipy')})
        result['code_sha256'] = hashlib.sha256(b''.join(p.name.encode()+p.read_bytes()
            for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()
        audits=[]
        for name, delta in (('neutral',0.),('interior_plus',.1),('interior_minus',-.1),('rotated',0.)):
            body.restore(original)
            if delta:
                for i,j in enumerate(body.active_joint_ids):
                    a,b = m.jnt_range[j]; q = int(m.jnt_qposadr[j])
                    d.qpos[q] = np.clip(q0[q]+delta*np.sin(i+1), a+1e-4,b-1e-4)
            if name=='rotated':
                d.qpos[:3] += [.3,-.2,.4]; d.qpos[3:7] = [.5,.5,.5,.5]
            mj.mj_forward(m,d)
            entry, matrices = audit_transmission(m,d)
            entry['pose']=name; audits.append(entry)
            for k,v in matrices.items(): arrays[f'{name}_{k}']=v
            arrays[name+'_qpos']=d.qpos.copy()
            if time.monotonic()>deadline:raise TimeoutError('Tendon audit budget exceeded')
        result['audits']=audits
        if not all(a['passed'] for a in audits):raise ValueError('Per-tendon force audit failed')
        body.restore(original)
        settle, settle_arrays = _trial(body,trans,original,fmax,l0,.15,'passive',None,deadline)
        result['passive_settle']=settle
        arrays.update({'settle_'+k:v for k,v in settle_arrays.items()})
        # All measured conditions start from the same settled geometry at zero velocity.
        d.qvel[:]=0.; d.qacc[:]=0.; d.qfrc_applied[:]=0.; mj.mj_forward(m,d)
        initial = body.state()
        limits = fmax*np.exp(-((d.ten_length/l0-1)/.5)**2)
        static, support_arrays = evaluate_support(body,trans,limits)
        arrays.update(support_arrays)
        result['static_support']=static
        trials=[]
        conditions=['passive','extensor_pulse','flexor_pulse','tonic_2pct']
        if static['feasible']:conditions.append('static_solution')
        for condition in conditions:
            trial, observations = _trial(body,trans,initial,fmax,l0,duration,condition,
                                           np.asarray(static.get('tensions_native',[])),deadline)
            trials.append(trial)
            arrays.update({condition+'_'+k:v for k,v in observations.items()})
        result['trials']=trials
        result['matched_initial_states']=all(np.array_equal(arrays[c+'_'+k][0],arrays['passive_'+k][0])
            for c in conditions for k in ('qpos','qvel'))
        if not result['matched_initial_states']:raise ValueError('Unmatched control initial state')
        # Benchmark only the force mapping at an identical frozen pose, not the full simulation.
        body.restore(original); tension=.01*fmax
        timings={}
        for name,fn,repeats in (('csr',lambda:trans.force(d,tension),100),
                                ('cartesian',lambda:__import__('organism_core.mechanics',fromlist=['pulling_force']).pulling_force(m,d,trans.paths,tension),5)):
            start=time.perf_counter()
            for _ in range(repeats):fn()
            timings[name+'_seconds_per_call']=(time.perf_counter()-start)/repeats
        result['force_mapping_benchmark']=timings
        result['status']='completed'
        result['engineering_audits_passed']=True
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        if arrays:
            path=out/'observations.npz'; np.savez_compressed(path,**arrays)
            with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
            result['observations']={'file':path.name,'sha256':digest}
        result['wall_seconds']=time.monotonic()-started
        write_result(out,result)
    return result
