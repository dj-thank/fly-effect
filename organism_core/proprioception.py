"""Six-leg receptor hypotheses, preserving real IDs and explicit uncertainty."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

TYPES={'SNpp50':'claw_flexion','SNpp51':'claw_extension',
       'SNpp39':'hook_flexion','SNpp41':'hook_extension',
       'SNpp40':'club','SNpp47':'club','SNpp59':'club','SNpp60':'club'}
LEGS=('lf','lm','lh','rf','rm','rh')
PRIORS={'status':'ASSUMED','max_rate_hz':100.,'position_width_rad':.25,
        'velocity_half_rad_s':5.,'adaptation_tau_s':.1,'position_baseline_hz':5.,
        'class_evidence':'Mamiya2018 qualitative position/direction/movement classes; MaleCNS type correspondence is not cell-specific physiology',
        'direction_assignment':'SNpp50/39 flexion; SNpp51/41 extension is an uncalibrated paired-sign hypothesis',
        'calcium_to_spikes_converted':False}

class Proprioceptors:
    def __init__(self,body):
        from .config import DATA
        root=DATA.parent
        path=DATA/'graph/neurons.parquet'
        self.annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        ann=pd.read_parquet(path)
        nerve={'ProLN':'f','MesoLN':'m','MetaLN':'h'}
        subset=ann[(ann.superclass=='vnc_sensory') & (ann.subclass=='chordotonal organ') &
                   ann.entryNerve.isin(nerve) & ann.rootSide.isin(['L','R']) & ann.type.isin(TYPES)].copy()
        subset=subset.sort_values('bodyId')
        self.ids=subset.bodyId.to_numpy(dtype=np.int64)
        self.leg=np.array([LEGS.index(side.lower()+nerve[n]) for side,n in zip(subset.rootSide,subset.entryNerve)])
        self.kind=np.array([TYPES[t] for t in subset.type])
        self.rows=[{'body_id':int(row.bodyId),'type':row.type,'leg':LEGS[int(self.leg[i])],
                    'response_hypothesis':str(self.kind[i]),'status':'inferred_class_and_direction_not_calibrated'} for i,row in enumerate(subset.itertuples())]
        if len(set(self.ids))!=len(self.ids) or set(self.leg)!=set(range(6)):
            raise ValueError('Six-leg sensory coverage missing or duplicated')
        joints=[next(i for i,name in enumerate(body.joint_names) if name.endswith(f'-{leg}_tibia-pitch')) for leg in LEGS]
        self.qpos=body.m.jnt_qposadr[joints].copy();self.dofs=body.m.jnt_dofadr[joints].copy()
        self.native_neutral=body.d.qpos[self.qpos].copy()
        self.neutral=np.array([body.joint_geometry[leg+':knee']['angle_neutral'] for leg in LEGS])
        self.extension_sign=np.array([body.joint_geometry[leg+':knee']['extension_sign'] for leg in LEGS])
        self.adaptation=np.zeros(6)
        self.last_rate=np.zeros(len(self.ids))
        self.counts=np.zeros(len(self.ids),dtype=np.int64)

    def rates(self,angles,velocities):
        angles=np.asarray(angles,dtype=float);velocities=np.asarray(velocities,dtype=float)
        if angles.shape!=(6,) or velocities.shape!=(6,) or not np.isfinite(np.r_[angles,velocities]).all():
            raise ValueError('Finite six-leg angle and velocity arrays required')
        p=PRIORS
        magnitude=np.abs(velocities)
        self.adaptation+=(magnitude-self.adaptation)*(-np.expm1(-.0001/p['adaptation_tau_s']))
        delta=(angles-self.neutral)[self.leg]
        velocity=velocities[self.leg]
        position=1/(1+np.exp(-np.clip(delta/p['position_width_rad'],-60,60)))
        base=p['position_baseline_hz'];maxrate=p['max_rate_hz'];half=p['velocity_half_rad_s']
        rates=np.zeros(len(self.ids))
        for kind,signal in (
            ('claw_flexion',base+(maxrate-base)*(1-position)),
            ('claw_extension',base+(maxrate-base)*position),
            ('hook_flexion',maxrate*np.maximum(-velocity,0)/(half+np.maximum(-velocity,0))),
            ('hook_extension',maxrate*np.maximum(velocity,0)/(half+np.maximum(velocity,0))),
            ('club',maxrate*np.abs(velocity)/(half+np.abs(velocity))/(1+self.adaptation[self.leg]/20.))):
            mask=self.kind==kind;rates[mask]=signal[mask]
        self.last_rate=np.clip(rates,0,maxrate)
        return self.last_rate.copy()

    def sample(self,body,rng,enabled=True):
        angles=self.neutral+self.extension_sign*(body.d.qpos[self.qpos]-self.native_neutral)
        rates=self.rates(angles,self.extension_sign*body.d.qvel[self.dofs])
        # Always consume the same RNG shape in input-cut controls.
        flags=rng.random(len(rates))<rates*.0001
        if not enabled:flags[:]=False
        self.counts+=flags
        return flags

    def state(self):
        return {'adaptation':self.adaptation.copy(),'last_rate':self.last_rate.copy(),'counts':self.counts.copy()}

    def restore(self,state):
        for key in ('adaptation','last_rate','counts'):
            value=state[key]
            if value.shape!=getattr(self,key).shape:raise ValueError('Receptor state shape mismatch')
            getattr(self,key)[:]=value
