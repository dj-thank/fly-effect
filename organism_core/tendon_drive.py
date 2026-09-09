"""Named muscle-template priors for the optional six-leg tendon drive."""
from pathlib import Path
import hashlib
import json
import numpy as np

# Mapping is a named-muscle transfer hypothesis; multi-compartment muscles share drive.
LABELS={
 'LFC_tergopleural_promotor_a':('Tergopleural/Pleural promotor',),
 'LFC_tergopleural_promotor_b':('Tergopleural/Pleural promotor',),
 'LFC_pleural_promotor':('Tergopleural/Pleural promotor',),
 'LFC_pleural_remotor_and_abductor':('Pleural remotor/abductor',),
 'LFC_sternal_anterior_rotator':('Sternal anterior rotator',),
 'LFC_sternal_posterior_rotator':('Sternal posterior rotator',),
 'LFC_sternal_adductor':('Sternal adductor',),
 'LFF_trochanter_flexor_a':('Tr flexor',),'LFF_trochanter_flexor_b':('Tr flexor',),
 'LFF_sterno-tergo-trochanter_extensor_a':('Sternotrochanter',),
 'LFF_sterno-tergo-trochanter_extensor_b':('Tergotr.',),
 'LFF_accesory_trochanter_flexor':('Acc. tr flexor',),
 'LFF_trochanter_extensor':('Tr extensor',),
 'LFTibia_flex_93434':('Ti flexor','Acc. ti flexor'),
 'LFTibia_extensor_93932':('Ti extensor',)}

PRIORS={'status':'ASSUMED model transfer','dt_s':.0001,'rate_tau_s':.02,'reference_rate_hz':100.,
        'activation_tau_s':.02,'deactivation_tau_s':.04,'force_length_width':.5,
        'maximum_shortening_lengths_s':5.,'optimal_length':'candidate neutral tendon length surrogate, not measured fiber length',
        'maximum_force':'source actuator gainprm[2]; force scale transferred without fitting walking',
        'unrepresented_LF_targets':'retain original lumped muscle path, separately counted'}

import mujoco as mj

class TendonDrive:
    def __init__(self,model,data,motor_rows,source_audit):
        import mujoco as mj
        audit=json.loads(Path(source_audit).read_text(encoding='utf-8'))
        if hashlib.sha256(Path(audit['source_path']).read_bytes()).hexdigest()!=audit['source_sha256']:raise ValueError('Changed muscle source')
        source={r['muscle']:r for r in audit['muscles']}
        self.names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_TENDON,i).removeprefix('candidate_').removesuffix('_tendon') for i in range(model.ntendon)]
        specs=[n.split('_',1) for n in self.names]
        if len(specs)!=90 or any(n not in LABELS for leg,n in specs):raise ValueError('Wrong six-leg tendon set')
        self.fmax=np.array([source[n]['source_gainprm'][2] for leg,n in specs]);self.l0=data.ten_length.copy()
        if (self.fmax<=0).any() or (self.l0<=0).any():raise ValueError('Invalid muscle parameter')
        self.assignments={};self.pool=np.zeros(len(specs));self.rows=[];lookup={r['body_id']:r for r in motor_rows}
        for row in motor_rows:
            if not row['required_for_walking']:continue
            leg=row['side'].lower()+{'T1':'f','T2':'m','T3':'h'}[row['neuromere']]
            donor=lookup[row['inferred_from_body_id']] if row['status']=='connectivity_inferred_mechanical_transfer' else row
            label=str(donor['type']).removesuffix(' MN')
            targets=[i for i,(l,n) in enumerate(specs) if l==leg and label in LABELS[n]]
            if targets:
                self.assignments[row['internal_index']]=targets;self.pool[targets]+=1
                self.rows.append({'body_id':row['body_id'],'internal_index':row['internal_index'],'type':row['type'],
                    'leg':leg,'muscles':[self.names[i] for i in targets],'basis':row['source_basis'],
                    'donor_body_id':donor['body_id'],'status':'LF muscle-template transfer hypothesis'})
        self.uninnervated=[n for n,p in zip(self.names,self.pool) if p==0];self.pool=np.maximum(self.pool,1)
        self.rate=np.zeros(len(specs));self.activation=np.zeros(len(specs));self.pending=np.zeros(len(specs))
        self.force=np.zeros(len(specs));self.events=0;self.saturated=np.zeros(len(specs),dtype=np.int64)

    def observe(self,indices,enabled=True):
        for index in map(int,indices):
            if index in self.assignments:
                self.events+=1
                if enabled:self.pending[self.assignments[index]]+=1

    def command(self,data):
        p=PRIORS;dt=p['dt_s'];self.rate*=np.exp(-dt/p['rate_tau_s'])
        self.rate+=self.pending/(self.pool*p['rate_tau_s']);self.pending[:]=0
        raw=self.rate/p['reference_rate_hz'];self.saturated+=raw>1;excitation=np.clip(raw,0,1)
        tau=np.where(excitation>self.activation,p['activation_tau_s'],p['deactivation_tau_s'])
        self.activation+=(excitation-self.activation)*(-np.expm1(-dt/tau))
        length=data.ten_length/self.l0;velocity=data.ten_velocity/self.l0;vmax=p['maximum_shortening_lengths_s']
        fl=np.exp(-((length-1)/p['force_length_width'])**2)
        shortened=np.minimum(velocity,0);elongated=np.maximum(velocity,0)
        fv=np.where(velocity<0,np.maximum(0,1+shortened/vmax)/(1-shortened/(.25*vmax)),1+.5*elongated/(elongated+.25*vmax))
        self.force=self.fmax*self.activation*fl*fv
        if not np.isfinite(self.force).all() or (self.force<0).any():raise ValueError('Invalid tendon force')
        return self.force.copy()


def pulling_force(model,data,tendons,tensions):
    generalized=np.zeros(model.nv)
    for sites,tension in zip(tendons,tensions):
        if tension==0:continue
        for a,b in zip(sites[:-1],sites[1:]):
            delta=data.site_xpos[b]-data.site_xpos[a];length=np.linalg.norm(delta)
            if length<=1e-12:raise ValueError('Degenerate tendon segment')
            force=float(tension)*delta/length
            for site,value in ((a,force),(b,-force)):
                mj.mj_applyFT(model,data,value,np.zeros(3),data.site_xpos[site],int(model.site_bodyid[site]),generalized)
    return generalized
