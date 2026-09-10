"""Bounded static force feasibility, NOT a controller or biological certificate."""
from __future__ import annotations
import numpy as np


def solve_support(tendon_map, contact_map, target, maximum_tension, friction):
    """Solve T f + C r = target, 0<=f<=fmax, rz>=0, |rx|+|ry|<=mu*rz.

The friction diamond is an inner approximation to a Coulomb cone. Infeasibility
is about this pose, these force limits and this approximation, not about whether
an animal/model could ever stand. Dry joint friction and limit forces are not
additional optimization variables. No neural or body parameter is modified.
"""
    from scipy.optimize import linprog
    T, C = np.asarray(tendon_map, dtype=float), np.asarray(contact_map, dtype=float)
    b, limit, mu = (np.asarray(x, dtype=float) for x in (target, maximum_tension, friction))
    if (T.ndim != 2 or C.ndim != 2 or T.shape[0] != C.shape[0] or not T.shape[0]
            or not T.shape[1] or C.shape[1] % 3 or not C.shape[1]
            or b.shape != (T.shape[0],) or limit.shape != (T.shape[1],)
            or mu.shape != (C.shape[1]//3,)):
        raise ValueError('Incompatible static support dimensions')
    if not all(np.isfinite(x).all() for x in (T, C, b, limit, mu)) or np.any(limit <= 0) or np.any(mu < 0):
        raise ValueError('Finite data, positive tension limits and nonnegative friction required')
    n = len(limit); k = len(mu); A = np.column_stack((T, C))
    # Scale equality rows, keeping differently dimensioned force/torque rows explicit.
    scale = np.maximum(np.abs(b), np.max(np.abs(T)*limit, axis=1))
    scale = np.maximum(scale, 1e-6)
    inequalities = np.zeros((4*k, n+3*k))
    for j in range(k):
        for r, (sx, sy) in enumerate(((1,1), (1,-1), (-1,1), (-1,-1))):
            inequalities[4*j+r, n+3*j:n+3*j+3] = [sx, sy, -mu[j]]
    bounds = [(0., float(x)) for x in limit] + [(None,None), (None,None), (0.,None)]*k
    result = linprog(np.r_[1/limit, np.zeros(3*k)], A_ub=inequalities,
                     b_ub=np.zeros(4*k), A_eq=A/scale[:,None], b_eq=b/scale,
                     bounds=bounds, method='highs', options={'time_limit': 10.})
    report = {'kind': 'static_force_feasibility', 'solver_status': int(result.status),
              'solver_message': str(result.message), 'solver_success': bool(result.success),
              'feasible': False, 'status': 'inconclusive', 'biological_validation': False, 'walking_claimed': False,
              'scope': 'One tested pose; bounded tensile forces and contact friction diamond only.'}
    if result.success:
        x = np.asarray(result.x, dtype=float)
        if x.shape != (A.shape[1],) or not np.isfinite(x).all():
            report['status'] = 'invalid_solver_solution'
            return report
        forces = x[n:].reshape(k,3)
        residual = A@x-b
        violations = [float(np.max(np.abs(residual)/scale)),
                      float(max(0., np.max(-x[:n]/limit))),
                      float(max(0., np.max(x[:n]/limit-1))),
                      float(max(0., np.max(-forces[:,2]))),
                      float(max(0., np.max(inequalities@x)))]
        report.update(feasible=all(v <= 1e-6 for v in violations),
                      tensions_native=x[:n].tolist(), contact_forces_native=forces.tolist(),
                      equality_residual_native=residual.tolist(), row_scales_native=scale.tolist(),
                      maximum_scaled_residual=violations[0], maximum_constraint_violation=max(violations[1:]),
                      maximum_tension_fraction=float(np.max(x[:n]/limit)))
        report['status'] = 'feasible' if report['feasible'] else 'residual_check_failed'
    elif result.status == 2:
        report['status'] = 'infeasible_under_declared_constraints'
        # Minimize normalized balance error; never relabel this as feasibility.
        normalized = A/scale[:,None]
        upper = np.vstack((np.column_stack((inequalities, np.zeros(len(inequalities)))),
                           np.column_stack((normalized, -np.ones(len(b)))),
                           np.column_stack((-normalized, -np.ones(len(b))))))
        nearest = linprog(np.r_[np.zeros(A.shape[1]), 1.], A_ub=upper,
                          b_ub=np.r_[np.zeros(len(inequalities)), b/scale, -b/scale],
                          bounds=bounds+[(0.,None)], method='highs', options={'time_limit':10.})
        report['nearest_balance_solver_status'] = int(nearest.status)
        if nearest.success and np.isfinite(nearest.x).all():
            residual = A@nearest.x[:-1]-b
            report['nearest_balance'] = {
                'is_feasible_solution': False,
                'minimum_scaled_balance_error': float(nearest.x[-1]),
                'scaled_residual': (residual/scale).tolist(),
                'residual_native': residual.tolist(),
                'tensions_native': nearest.x[:n].tolist(),
                'contact_forces_native': nearest.x[n:-1].reshape(k,3).tolist()}
    return report
