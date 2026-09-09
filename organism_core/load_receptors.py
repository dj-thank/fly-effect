"""Explicit uncalibrated foot-load surrogate for real leg TrCS candidates."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

LEGS=('lf','lm','lh','rf','rm','rh')
PRIORS={'status':'ASSUMED load-to-spike surrogate, not calibrated','dt_s':.0001,'filter_tau_s':.01,
        'maximum_rate_hz':100.,'half_rate_force':'body weight divided by six in native force units',
        'measurement':'magnitude of summed world ground-reaction force on each foot, previous completed physics interval',
        'two_cells_per_leg':'same scalar load hypothesis with independent Poisson samples; orientations unknown',
        'classification_source':'https://elifesciences.org/reviewed-preprints/97766v1',
        'classification_scope':'SNpp53 candidate trochanter campaniform sensilla; local MaleCNS IDs from locked annotations, not MANC IDs'}

class LoadReceptors:
    def __init__(self,body):
        from .config import DATA
        path=DATA/'graph/neurons.parquet'
        self.annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        ann=pd.read_parquet(path);nerves={'ProLN':'f','MesoLN':'m','MetaLN':'h'}
        x=ann[(ann.superclass=='vnc_sensory')&(ann.subclass=='campaniform sensilla')&(ann.type=='SNpp53')&ann.entryNerve.isin(nerves)&ann.rootSide.isin(['L','R'])].sort_values('bodyId')
        self.ids=x.bodyId.to_numpy(dtype=np.int64)
        self.leg=np.array([LEGS.index(side.lower()+nerves[nerve]) for side,nerve in zip(x.rootSide,x.entryNerve)])
        if len(self.ids)!=12 or not np.array_equal(np.bincount(self.leg,minlength=6),[2]*6):raise ValueError('Unexpected leg load receptor coverage')
        self.half_force=float(body.m.body_mass.sum()*abs(body.m.opt.gravity[2])/6)
        if self.half_force<=0:raise ValueError('Positive body weight required')
        self.filtered=np.zeros(6);self.last_force=np.zeros((6,3));self.last_rate=np.zeros(12);self.counts=np.zeros(12,dtype=np.int64)

    def rates(self,forces):
        forces=np.asarray(forces,dtype=float)
        if forces.shape!=(6,3) or not np.isfinite(forces).all():raise ValueError('Finite per-foot world force vectors required')
        self.last_force[:]=forces;load=np.linalg.norm(forces,axis=1)
        self.filtered+=(load-self.filtered)*(-np.expm1(-PRIORS['dt_s']/PRIORS['filter_tau_s']))
        scalar=PRIORS['maximum_rate_hz']*self.filtered/(self.half_force+self.filtered)
        self.last_rate=scalar[self.leg]
        return self.last_rate.copy()

    def sample(self,body,rng,enabled=True,forces=None):
        rates=self.rates(body.foot_loads() if forces is None else forces)
        flags=rng.random(len(self.ids))<rates*PRIORS['dt_s']
        if not enabled:flags[:]=False
        self.counts+=flags
        return flags

    def state(self):
        return {k:getattr(self,k).copy() for k in ('filtered','last_force','last_rate','counts')}

    def restore(self,state):
        for key,current in self.state().items():
            value=state[key]
            if value.shape!=current.shape or value.dtype!=current.dtype:raise ValueError('Load state shape/type mismatch')
            getattr(self,key)[:]=value
