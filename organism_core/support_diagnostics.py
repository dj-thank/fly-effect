"""Flat-ground, constraint-active support diagnostics; not a gait certificate."""
from __future__ import annotations
import numpy as np
import mujoco as mj
from .contacts import is_active_contact
from .static_support import solve_support
from .force_audit import require_unforced_static_state

LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
CONTACT_SEMANTICS = 'constraint_active: efc_address >= 0 and exclude == 0'


def contact_columns(model, data):
    """Use solver-active contacts, including positive-distance margins.

Forces are expressed in WORLD xyz with compression along +z. Sloped ground
is explicitly unsupported; frictionless contacts have zero tangential capacity.
Spin/rolling moments are not optimization variables. Source data is not mutated.
"""
    blocks, friction, records, nonfoot = [], [], [], []
    for contact in data.contact[:data.ncon]:
        if not is_active_contact(contact):
            continue
        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(g)) or ''
                 for g in (contact.geom1, contact.geom2)]
        if 'ground_plane' not in names:
            continue
        other = 1 if names[0] == 'ground_plane' else 0
        name = names[other]
        if not any(f'/{leg}_tarsus' in name for leg in LEGS):
            nonfoot.append(name)
            continue
        normal = np.asarray(contact.frame).reshape(3, 3)[0] * (1 if other == 1 else -1)
        if not np.allclose(normal, [0., 0., 1.], rtol=0., atol=1e-7):
            raise ValueError('Static support requires horizontal ground with upward normal')
        geom = int((contact.geom1, contact.geom2)[other])
        jac = np.zeros((3, model.nv))
        mj.mj_jac(model, data, jac, None, contact.pos, int(model.geom_bodyid[geom]))
        blocks.append(jac.T)
        mu = 0. if int(contact.dim) == 1 else float(min(contact.friction[:2]))
        friction.append(mu)
        records.append({'geom': name, 'position_native': contact.pos.tolist(), 'mu': mu,
                        'distance_native': float(contact.dist),
                        'includemargin_native': float(contact.includemargin),
                        'efc_address': int(contact.efc_address), 'condim': int(contact.dim)})
    matrix = np.column_stack(blocks) if blocks else np.zeros((model.nv, 0))
    return matrix, np.asarray(friction), records, sorted(set(nonfoot))


def evaluate_support(body, transmission, limits):
    """Return a scoped result AND reproducible LP inputs at the current pose.

Call after mj_forward with zero qvel. No controller or biological status is set.
Ineligible geometry, proved LP infeasibility and solver failure stay distinct.
"""
    m, d = body.m, body.d
    require_unforced_static_state(d)
    C, mu, contacts, nonfoot = contact_columns(m, d)
    T = transmission.matrix(d)
    target = d.qfrc_bias - d.qfrc_passive
    arrays = {'support_tendon_map': T, 'support_contact_map': C,
              'support_target': target.copy(), 'support_limits': np.asarray(limits).copy(),
              'support_friction': mu, 'support_qpos': d.qpos.copy()}
    feet, reaction = body.contacts()
    column_feet = {leg: any(f'/{leg}_tarsus' in r['geom'] for r in contacts) for leg in LEGS}
    if feet != column_feet:
        raise ValueError('Body stance flags disagree with support contact columns')
    result = {'kind': 'static_force_feasibility', 'feasible': False,
              'status': 'not_eligible', 'contact_semantics': CONTACT_SEMANTICS,
              'foot_contacts': contacts, 'nonfoot_contacts': nonfoot,
              'contact_flags_match': True, 'feet': feet,
              'ground_reaction_at_probe_native': reaction.tolist(),
              'biological_validation': False, 'walking_claimed': False}
    if not contacts:
        result['reason'] = 'no_active_foot_contacts'
    elif nonfoot:
        result['reason'] = 'active_nonfoot_contacts'
    elif not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError('Finite positive support tension limits required')
    else:
        result.update(solve_support(T, C, target, limits, mu))
        if result['status'] == 'feasible':
            result['status'] = 'feasible_at_tested_pose'
        if 'nearest_balance' in result:
            scaled = result['nearest_balance']['scaled_residual']
            order = np.argsort(np.abs(scaled))[::-1][:6]
            result['largest_balance_residuals'] = [
                {'dof': int(i), 'joint': mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT,
                                                     int(m.dof_jntid[i])),
                 'scaled_residual': float(scaled[i]),
                 'native_residual': result['nearest_balance']['residual_native'][i]}
                for i in order]
    return result, arrays
