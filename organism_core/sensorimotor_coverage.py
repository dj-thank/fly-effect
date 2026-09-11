"""Runtime mapping census, not anatomical or walking validation.

Count actual MotorMap assignments and actual sensory IDs independently of the
full-CNS neuron registry. Keep unmapped walking motors separate from motors
outside the current walking scope; check conservation of every motor event.
"""
from __future__ import annotations

from collections import Counter
import numpy as np
from .connectivity_audit import ratio, _registry


def summarize(ids, motor_indices, rows, assignments, sensory_ids, accounting, *, joint_names, motor_spikes=None):
    ids, motor_mask, _ = _registry(ids, motor_indices, np.zeros(len(ids), dtype=np.int8))
    indices = set(map(int, np.flatnonzero(motor_mask)))
    if not isinstance(rows, list) or not isinstance(assignments, dict):
        raise ValueError('Motor rows and actual assignments are required')
    if (not isinstance(joint_names, (list, tuple)) or any(not isinstance(j, str) or not j for j in joint_names)
            or len(set(joint_names)) != len(joint_names)):
        raise ValueError('Unique joint names are required')
    by_index = {}
    for row in rows:
        index = row.get('internal_index') if isinstance(row, dict) else None
        if type(index) is not int or index not in indices or index in by_index:
            raise ValueError('Unknown or duplicate motor mapping row')
        if type(row.get('body_id')) is not int or row['body_id'] != int(ids[index]):
            raise ValueError('Motor row body ID does not match graph registry')
        if type(row.get('required_for_walking')) is not bool or type(row.get('anatomically_calibrated')) is not bool:
            raise ValueError('Motor scope and anatomical flags must be booleans')
        if not isinstance(row.get('status'), str) or not row['status']:
            raise ValueError('Explicit motor mapping status required')
        by_index[index] = row
    if set(by_index) != indices:
        raise ValueError('Motor mapping rows must cover the entire motor registry')
    required = {i for i, r in by_index.items() if r['required_for_walking']}
    mapped = set(assignments)
    if any(type(i) is not int for i in mapped) or not mapped <= required:
        raise ValueError('Actual assignments must belong to walking-scope motors')
    for i, row in by_index.items():
        if (row.get('joint') is not None) != (i in mapped):
            raise ValueError('Reported joint and actual assignment disagree')
        if i in mapped:
            value = assignments[i]
            if (not isinstance(value, tuple) or len(value) != 2
                    or any(type(x) is not int for x in value)
                    or not 0 <= value[0] < len(joint_names) or value[1] not in (0, 1)):
                raise ValueError('Invalid joint/polarity assignment')
            if row['joint'] != joint_names[value[0]]:
                raise ValueError('Joint name and actual assignment disagree')
    sensory = np.asarray(sensory_ids)
    if (sensory.ndim != 1 or sensory.dtype.kind not in 'iu'
            or len(np.unique(sensory)) != len(sensory)
            or not np.isin(sensory, ids).all()):
        raise ValueError('Sensory IDs must be unique registered body IDs')
    if np.isin(sensory, ids[motor_mask]).any():
        raise ValueError('Sensory IDs overlap motor registry')
    if not isinstance(accounting, dict) or not isinstance(accounting.get('per_motor_spikes'), dict):
        raise ValueError('Per-motor event accounting is required')
    per_motor = accounting['per_motor_spikes']
    if set(per_motor) != {str(i) for i in indices}:
        raise ValueError('Per-motor event accounting IDs mismatch')
    if any(type(v) is not int or v < 0 for v in per_motor.values()):
        raise ValueError('Motor event counts must be nonnegative integers')
    populations = {'mapped_spikes': mapped, 'unresolved_spikes': required - mapped,
                   'outside_spikes': indices - required}
    events = {}
    for name, population in populations.items():
        value = accounting.get(name)
        if type(value) is not int or value < 0 or value != sum(per_motor[str(i)] for i in population):
            raise ValueError('Motor event conservation failed: ' + name)
        events[name] = value
    total = sum(per_motor.values())
    if motor_spikes is not None and (type(motor_spikes) is not int or motor_spikes != total):
        raise ValueError('Motor map event total differs from neural spike monitor')
    calibrated = sum(by_index[i]['anatomically_calibrated'] for i in mapped)
    active = sum(v > 0 for v in per_motor.values())
    return {'schema': 1, 'kind': 'runtime_mapping_not_biological_validation',
            'registered_neurons': len(ids), 'motor_registry': ratio(len(indices), len(ids)),
            'nonmotor_registry': ratio(len(ids) - len(indices), len(ids)),
            'walking_scope': ratio(len(required), len(indices)),
            'outside_walking_scope': ratio(len(indices - required), len(indices)),
            'mapped_walking_motors': ratio(len(mapped), len(required)),
            'unresolved_walking_motors': ratio(len(required - mapped), len(required)),
            'anatomically_calibrated_mapped': ratio(calibrated, len(mapped)),
            'body_input_neurons': ratio(len(sensory), len(ids)),
            'active_motor_neurons': ratio(active, len(indices)),
            'motor_events': {'total': total, **events, 'conservation_verified': True},
            'mapping_status_counts': dict(sorted(Counter(str(r.get('status', 'unspecified')) for r in rows).items())),
            'biological_validation': False, 'walking_claimed': False,
            'notes': ['Mapping to a joint is not identification of its biological muscle insertion.',
                      'Mapped motor events are not proof of force, support, or walking.',
                      'Sensory registry fraction is not coverage of all biological sensory neurons.',
                      'Nonmotor membership is not a claim of being unrelated to movement.']}
