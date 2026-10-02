"""Classify retained FE-02 search records without a physics backend.

These rules leave schema-1 evidence and frozen search protocols unchanged.
A verified witness establishes existence even when other work was cut short;
a negative search outcome requires the declared work to have finished.
"""
from __future__ import annotations

import math


INFEASIBLE = 'infeasible_under_declared_constraints'
FEASIBLE_POSE = 'feasible_at_tested_pose'
COMPLETE_DESCENTS = {'step_floor_reached', 'residual_zero_live'}
CLASSIFICATION_VERSION = 2


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _certifications(records):
    certs = [c for r in records for c in r.get('certifications', [])]
    completed = [c for c in certs if c.get('status') == 'completed']
    samples = [s for c in completed for s in c.get('samples', [])
               if s.get('evidence_saved') is True]
    errors = []
    for index, cert in enumerate(completed):
        saved = [s for s in cert.get('samples', [])
                 if s.get('evidence_saved') is True]
        for key, expected in (
                ('any_full_feasible', any(s.get('support', {}).get('status')
                                         == FEASIBLE_POSE for s in saved)),
                ('any_passive_feasible', any(s.get('passive_subsystem', {}).get('status')
                                            == 'feasible' for s in saved))):
            if type(cert.get(key)) is not bool or cert[key] != expected:
                errors.append(f'certification {index}: {key} disagrees with saved samples')
    undecided = any(
        s.get('support', {}).get('status') not in (FEASIBLE_POSE, INFEASIBLE)
        or s.get('passive_subsystem', {}).get('status') not in ('feasible', INFEASIBLE)
        or s.get('margin', {}).get('status') not in ('measured', 'unreachable_at_any_shift')
        for s in samples)
    # A failed root gate may have no full-support arrays. Numerical failure
    # still cannot be treated as a decided negative just because it was skipped.
    undecided = undecided or any(
        s.get('evidence_saved') is not True and 'root_balance' in s
        and s['root_balance'].get('status') not in ('not_eligible', INFEASIBLE)
        for c in completed for s in c.get('samples', []))
    return completed, samples, undecided, errors


def _descent_stops(records, protocol, certification_limit):
    reasons = set()
    for record in records:
        status = record.get('status')
        if status not in COMPLETE_DESCENTS:
            reasons.add('descent:' + str(status or 'not_run'))
        if record.get('unresolved_evals', 0):
            reasons.add('unresolved_descent_evaluations')
        if ('initial_status' in record and record['initial_status'] not in
                ('feasible', 'infeasible', 'residual_check_failed', 'no_eligible_sample')):
            reasons.add('unresolved_descent_evaluations')
        certs = record.get('certifications', [])
        if not certs:
            reasons.add('certifications_unexecuted')
        for cert in certs:
            if cert.get('status') != 'completed':
                reasons.add('certification:' + str(cert.get('status') or 'not_run'))
            elif not cert.get('samples'):
                reasons.add('certification_samples_missing')
            else:
                samples = cert['samples']
                depths = protocol['depths_native']
                if (len(samples) != len(depths)
                        or {s.get('index') for s in samples} != set(range(len(depths)))
                        or any(s.get('depth_native') != depths[s['index']]
                               for s in samples if type(s.get('index')) is int
                               and 0 <= s['index'] < len(depths))):
                    reasons.add('certification_depths_incomplete')
        last = record.get('accepted_steps')
        expected = {last}
        initial, final = record.get('initial_R'), record.get('final_R')
        if type(last) is int and last > 1 and _finite(initial) and _finite(final):
            accepted = [a for a in record.get('accepted', []) if _finite(a.get('R'))]
            mid = min(accepted, key=lambda a: abs(a['R']-(initial+final)/2),
                      default={'accepted': last})['accepted']
            expected.add(mid)
        expected = set(sorted(expected, reverse=True)[:certification_limit])
        if last is None or {c.get('accepted_index') for c in certs} != expected:
            reasons.add('certifications_incomplete')
    return reasons


def _outcome(samples, tolerance_witness, improved, negative, reasons, undecided, errors):
    if undecided:
        reasons.add('undecided_saved_evidence')
    # Positives come from individual saved witnesses, never cached summary flags.
    if errors:
        outcome = 'invalid_experiment'
    elif any(s.get('support', {}).get('status') == FEASIBLE_POSE for s in samples):
        outcome = 'found_full_support_pose'
    elif any(s.get('passive_subsystem', {}).get('status') == 'feasible' for s in samples):
        outcome = 'found_subsystem_feasible_pose'
    elif tolerance_witness:
        outcome = 'tolerance_support_witness'
    elif improved:
        outcome = 'global_basin_improvement'
    elif reasons:
        outcome = 'inconclusive'
    else:
        outcome = negative
    return {'scientific_outcome': outcome,
            'outcome_classification': {'version': CLASSIFICATION_VERSION,
                                       'search_complete': not reasons and not errors,
                                       'stop_reasons': sorted(reasons)},
            'errors': errors}


def full_descent_outcome(result, protocol, *, replay_undecided=False,
                         tolerance_witness=None):
    """Use the same stopping/witness rules for a run and saved-input replay."""
    seeds = result.get('seeds', [])
    completed = [s for s in seeds if s.get('status') == 'completed']
    descents = [dict(s.get('descent', {}), certifications=s.get('certifications', []))
                for s in completed]
    reasons = _descent_stops(descents, protocol, protocol['certified_poses_per_seed'])
    if result.get('status') == 'timeout':
        reasons.add('worker_wall_budget_exhausted')
    declared = result.get('declared_seeds', 0)
    if not declared:
        reasons.add('no_declared_seeds')
    if len(completed) != declared or any(s.get('remaining_seeds_unexecuted') for s in seeds):
        reasons.add('seeds_unexecuted')
    _, samples, undecided, errors = _certifications(descents)
    if tolerance_witness is None:
        tol = protocol['support_residual_tolerance']
        tolerance_witness = any(_finite(a.get('R')) and 0 <= a['R'] <= tol
                                for d in descents for a in d.get('accepted', []))
    return _outcome(samples, tolerance_witness, False,
                    'descent_floored_without_feasibility', reasons,
                    undecided or replay_undecided, errors)


def global_sample_outcome(result, protocol, *, replay_undecided=False,
                          tolerance_witness=None, basin_improved=None):
    """A skipped or interrupted top-k descent cannot justify a negative."""
    samples = result.get('samples', [])
    evaluated = [s for s in samples if s.get('status') == 'evaluated']
    eligible = sorted((s for s in evaluated if s.get('eligible') is True
                       and _finite(s.get('R'))), key=lambda s: s['R'])
    selected = eligible[:protocol['descend_top_k']]
    descents = [s.get('descent', {}) for s in selected]
    reasons = _descent_stops(descents, protocol, protocol['certified_poses_per_sample'])
    if result.get('status') == 'timeout':
        reasons.add('worker_wall_budget_exhausted')
    if any(s.get('evidence_saved') is True and (
            s.get('probe_status') not in ('feasible', 'infeasible', 'residual_check_failed')
            or not _finite(s.get('R'))) for s in evaluated):
        reasons.add('undecided_saved_probe')
    if len(evaluated) != protocol['n_samples'] or any(
            s.get('remaining_samples_unexecuted') for s in samples):
        reasons.add('samples_unexecuted')
    _, certified, undecided, errors = _certifications(descents)
    if tolerance_witness is None:
        tol = protocol['support_residual_tolerance']
        residuals = [s.get('R') for s in evaluated if s.get('evidence_saved') is True]
        residuals += [a.get('R') for d in descents for a in d.get('accepted', [])]
        tolerance_witness = any(_finite(r) and 0 <= r <= tol for r in residuals)
    if basin_improved is None:
        baseline = result.get('baseline_floor')
        basin_improved = _finite(baseline) and any(
            _finite(d.get('final_R')) and 0 <= d['final_R'] < baseline for d in descents)
    return _outcome(certified, tolerance_witness, basin_improved,
                    'no_global_improvement', reasons,
                    undecided or replay_undecided, errors)


def full_descent_baseline(result, verification):
    """Return the best observed terminal residual, without claiming a floor.

    Old schema-1 observations can be replayed with the current classifier;
    their original outcome text is not accepted as a current verification.
    """
    if (verification.get('evidence_valid') is not True
            or verification.get('outcome_classification', {}).get('version')
            != CLASSIFICATION_VERSION):
        raise ValueError('Full-descent evidence needs current outcome re-verification')
    return observed_terminal_residual(result)


def observed_terminal_residual(result):
    """The minimum recorded terminal residual is an observation, not a floor."""
    residuals = [s.get('descent', {}).get('final_R') for s in result.get('seeds', [])
                 if s.get('status') == 'completed']
    finite = [r for r in residuals if _finite(r) and r >= 0]
    if not finite:
        raise ValueError('Full-descent evidence has no finite observed terminal residual')
    return min(finite)
