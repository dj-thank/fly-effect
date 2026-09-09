"""Explicit named-target transfer; uncertain IDs remain visible and unassigned."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

# Axis/sign choices are engineering hypotheses, NOT identified muscle insertions.
# The scalar sign describes native joint angle; anatomy calibration remains pending.
TARGETS={
 'Ti flexor':('tibia','pitch',1),'Acc. ti flexor':('tibia','pitch',1),
 'Ti extensor':('tibia','pitch',-1),'Tr flexor':('trochanterfemur','pitch',-1),
 'Acc. tr flexor':('trochanterfemur','pitch',-1),'Tr extensor':('trochanterfemur','pitch',1),
 'Sternal anterior rotator':('coxa','yaw',1),'Sternal posterior rotator':('coxa','yaw',-1),
 'Pleural remotor/abductor':('coxa','roll',1),'Sternal adductor':('coxa','roll',-1),
 'Tergopleural/Pleural promotor':('coxa','pitch',1),
 'Sternotrochanter':('coxa','pitch',-1),'Tergotr.':('coxa','pitch',-1),
 'Fe reductor':('trochanterfemur','roll',1),
 'Ta depressor':('tarsus1','pitch',1),'Ta levator':('tarsus1','pitch',-1),
 'ltm':('tarsus1','pitch',1),'ltm1-tibia':('tarsus1','pitch',1),'ltm2-femur':('tarsus1','pitch',1),
}

class MotorMap:
    def __init__(self,ids,motor_indices,joint_names,inference_path=None):
        from .config import DATA
        root=DATA.parent
        annotation=DATA/'graph/neurons.parquet'
        table=DATA/'manc-supplements/elife-96084-supp3-v1.csv'
        self.source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (annotation,table)}
        ann=pd.read_parquet(annotation).set_index('bodyId',verify_integrity=True)
        supplement=pd.read_csv(table)
        groups={int(g):set(t.dropna().astype(str)) for g,t in supplement.groupby('group').target}
        self.rows=[];self.assignments={};self.pool_sizes=np.zeros((len(joint_names),2))
        self.motor_set=set(map(int,motor_indices));self.mapped_set=set()
        for index in motor_indices:
            body_id=int(ids[index]);row=ann.loc[body_id]
            record={'body_id':body_id,'internal_index':int(index),'type':row['type'],
                    'subclass':row['subclass'],'side':row['somaSide'],'neuromere':row['somaNeuromere'],
                    'required_for_walking':row['subclass'] in ('fl','ml','hl'),
                    'status':'outside_current_walking_scope','source_basis':None,'joint':None,
                    'muscle_polarity':None,'anatomically_calibrated':False}
            if record['required_for_walking']:
                label=str(row['type']).removesuffix(' MN')
                curated=groups.get(int(row['mancGroup']),set()) if pd.notna(row['mancGroup']) else set()
                record['curated_targets']=sorted(curated)
                record['status']='unresolved_target'
                if label in TARGETS and (not curated or curated=={label}):
                    segment,axis,sign=TARGETS[label]
                    leg=str(row['somaSide']).lower()+{'T1':'f','T2':'m','T3':'h'}.get(row['somaNeuromere'],'?')
                    matches=[i for i,n in enumerate(joint_names) if n.endswith(f'-{leg}_{segment}-{axis}')]
                    if len(matches)==1:
                        j=matches[0];polarity=0 if sign>0 else 1
                        self.assignments[int(index)]=(j,polarity)
                        self.pool_sizes[j,polarity]+=1;self.mapped_set.add(int(index))
                        record.update(status='inferred_mechanical_transfer',source_basis='curated_group_and_type' if curated else 'MaleCNS_named_type',
                                      joint=joint_names[j],muscle_polarity=sign)
                elif any(x in TARGETS for x in curated) and curated!={label}:record['status']='curated_type_conflict'
            self.rows.append(record)
        if inference_path is not None:
            inference_path=Path(inference_path)
            inference=json.loads(inference_path.read_text(encoding='utf-8'))
            if inference['source_hashes']!=self.source_hashes:raise ValueError('Inference source data changed')
            if inference['graph_sha256']!='45233dbc6a67b676bdf5a6b1ee0fc206bea5eae66074f92a09efcbb56aef89d5' or inference['minimum_cosine']!=.1:
                raise ValueError('Inference contract mismatch')
            known={r['body_id']:r for r in self.rows if r['joint'] is not None}
            by_id={r['body_id']:r for r in self.rows}
            seen=set()
            for item in inference['neurons']:
                if item['body_id'] in seen:raise ValueError('Duplicate inferred neuron')
                seen.add(item['body_id']);record=by_id[item['body_id']]
                record['connectivity_prior_candidates']=item['candidates']
                if record['joint'] is not None or not item['usable_as_candidate_prior']:continue
                best=item['candidates'][0]
                if not .1<=best['cosine']<=1.00000001:raise ValueError('Invalid connectivity confidence')
                donor=known[best['body_id']]
                if donor['side']!=record['side'] or donor['neuromere']!=record['neuromere']:raise ValueError('Cross-leg prior forbidden')
                if donor['joint']!=best['joint'] or donor['muscle_polarity']!=best['muscle_polarity']:raise ValueError('Mechanical target changed')
                j=joint_names.index(donor['joint']);polarity=0 if donor['muscle_polarity']>0 else 1
                self.assignments[record['internal_index']]=(j,polarity)
                self.pool_sizes[j,polarity]+=1;self.mapped_set.add(record['internal_index'])
                record.update(status='connectivity_inferred_mechanical_transfer',source_basis='same_leg_incoming_connectivity_prior',
                              joint=donor['joint'],muscle_polarity=donor['muscle_polarity'],inferred_from_body_id=donor['body_id'])
            self.source_hashes[str(inference_path.resolve())]=hashlib.sha256(inference_path.read_bytes()).hexdigest()
        self.pool_sizes=np.maximum(self.pool_sizes,1)
        self.outside_spikes=0;self.unresolved_spikes=0;self.mapped_spikes=0
        self.per_motor_spikes={int(i):0 for i in motor_indices}

    def decode(self,fired):
        counts=np.zeros_like(self.pool_sizes)
        for index in map(int,fired):
            if index not in self.motor_set:continue
            self.per_motor_spikes[index]+=1
            if index in self.assignments:
                j,pol=self.assignments[index];counts[j,pol]+=1;self.mapped_spikes+=1
            else:
                rec=next(r for r in self.rows if r['internal_index']==index)
                if rec['required_for_walking']:self.unresolved_spikes+=1
                else:self.outside_spikes+=1
        return counts

    def state(self):
        return {'outside_spikes':self.outside_spikes,'unresolved_spikes':self.unresolved_spikes,
                'mapped_spikes':self.mapped_spikes,'per_motor_spikes':{str(k):v for k,v in self.per_motor_spikes.items()}}

    def restore(self,state):
        for key in ('outside_spikes','unresolved_spikes','mapped_spikes'):setattr(self,key,state[key])
        self.per_motor_spikes={int(k):v for k,v in state['per_motor_spikes'].items()}
