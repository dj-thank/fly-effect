"""Single-accounting hybrid of registered tendons and remaining lumped muscles."""
from xml.etree import ElementTree as ET
import numpy as np
import mujoco as mj
from .tendon_drive import TendonDrive,pulling_force

class HybridMuscles:
    def __init__(self,body,mapping,source_audit):
        self.body=body;self.legacy=body.muscles
        mj.mj_forward(body.m,body.d)
        self.tendons=TendonDrive(body.m,body.d,mapping.rows,source_audit)
        self.migrated=set(self.tendons.assignments)
        pool=np.zeros_like(mapping.pool_sizes)
        for index,(joint,polarity) in mapping.assignments.items():
            if index not in self.migrated:pool[joint,polarity]+=1
        self.legacy.pool_sizes=np.maximum(pool,1)
        self.sites=[[mj.mj_name2id(body.m,mj.mjtObj.mjOBJ_SITE,e.get('site')) for e in t] for t in ET.fromstring(body.xml).find('tendon')]
        self.last_torque=np.zeros(len(body.active_dofs));self.max_base_residual=0.

    @property
    def activation(self):return np.r_[self.legacy.activation.ravel(),self.tendons.activation]
    @property
    def rate(self):return np.r_[self.legacy.rate.ravel(),self.tendons.rate]
    @property
    def pending(self):return np.r_[self.legacy.pending.ravel(),self.tendons.pending]
    @property
    def fiber_force(self):return np.r_[self.legacy.fiber_force.ravel(),self.tendons.force]

    def command(self,q,velocity):
        b=self.body
        mj.mj_fwdPosition(b.m,b.d);mj.mj_fwdVelocity(b.m,b.d)
        tension=self.tendons.command(b.d);legacy=self.legacy.command(q,velocity)
        force=pulling_force(b.m,b.d,self.sites,tension)
        self.max_base_residual=max(self.max_base_residual,float(np.max(np.abs(force[:6]))))
        if self.max_base_residual>1e-10:raise ValueError('Tendon applies external base force')
        self.last_torque=legacy+force[b.active_dofs]
        return self.last_torque.copy()

    def observe_neurons(self,indices,mapping):
        fallback=mapping.decode(indices) # Count every real motor event exactly once.
        for index in map(int,indices):
            if index in self.migrated:
                joint,polarity=mapping.assignments[index];fallback[joint,polarity]-=1
        self.legacy.observe(fallback);self.tendons.observe(indices)

    def state(self):
        return {'legacy':self.legacy.state(),
                'tendons':{k:getattr(self.tendons,k).copy() for k in ('rate','activation','pending','force','saturated')},
                'events':self.tendons.events,'last_torque':self.last_torque.copy(),'max_base_residual':self.max_base_residual}

    def restore(self,state):
        self.legacy.restore(state['legacy'])
        for key in ('rate','activation','pending','force','saturated'):
            value=state['tendons'][key];current=getattr(self.tendons,key)
            if value.shape!=current.shape or value.dtype!=current.dtype:raise ValueError('Tendon state shape/type mismatch')
            current[:]=value
        if state['last_torque'].shape!=self.last_torque.shape:raise ValueError('Torque state shape mismatch')
        self.tendons.events=state['events'];self.last_torque[:]=state['last_torque'];self.max_base_residual=state['max_base_residual']
