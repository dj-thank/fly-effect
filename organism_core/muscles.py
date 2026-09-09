"""Antagonistic lumped muscles driven by spike counts, not target joint angles.

All numerical parameters below are declared engineering priors. This is not an
empirically calibrated neuromuscular model. Length is mm and force uses the
body model's native force unit until the full unit audit is complete.
"""
import numpy as np

PRIORS={'status':'ASSUMED','dt_s':.0001,'activation_tau_s':.02,'deactivation_tau_s':.04,
        'rate_tau_s':.02,'reference_rate_hz':100.,'fiber_length_mm':.5,
        'moment_arm_mm':.05,'maximum_force_native':100.,'maximum_shortening_lengths_s':5.,
        'force_length_width':.5,'reason':'Initial bounded muscle-unit approximation; fit only in isolated muscle/kinematic calibration'}


class MuscleBank:
    def __init__(self,neutral_joint_angles,pool_sizes=None):
        self.q0=np.asarray(neutral_joint_angles,dtype=float).copy()
        if self.q0.ndim!=1 or not np.isfinite(self.q0).all():raise ValueError('Finite joint coordinates required')
        n=len(self.q0)
        self.activation=np.zeros((n,2));self.rate=np.zeros((n,2));self.pending=np.zeros((n,2))
        self.pool_sizes=np.ones((n,2)) if pool_sizes is None else np.asarray(pool_sizes,dtype=float)
        if self.pool_sizes.shape!=(n,2) or (self.pool_sizes<1).any():raise ValueError('Nonempty pool denominators required')
        self.saturated=np.zeros((n,2),dtype=np.int64)
        self.fiber_force=np.zeros((n,2));self.last_torque=np.zeros(n)
        self.work_native=0.;self.tick=0

    def command(self,q,velocity):
        q=np.asarray(q,dtype=float);velocity=np.asarray(velocity,dtype=float)
        if q.shape!=self.q0.shape or velocity.shape!=q.shape or not np.isfinite(np.r_[q,velocity]).all():
            raise ValueError('Invalid muscle geometry observation')
        p=PRIORS;dt=p['dt_s']
        self.rate*=np.exp(-dt/p['rate_tau_s'])
        self.rate+=self.pending/(self.pool_sizes*p['rate_tau_s']);self.pending[:]=0
        raw=self.rate/p['reference_rate_hz'];self.saturated+=raw>1
        excitation=np.clip(raw,0,1)
        tau=np.where(excitation>self.activation,p['activation_tau_s'],p['deactivation_tau_s'])
        self.activation+=(excitation-self.activation)*(-np.expm1(-dt/tau))
        sign=np.array([1.,-1.])
        length=1-(q-self.q0)[:,None]*sign*p['moment_arm_mm']/p['fiber_length_mm']
        shortening=-velocity[:,None]*sign*p['moment_arm_mm']/p['fiber_length_mm']
        fl=np.exp(-((length-1)/p['force_length_width'])**2)
        vmax=p['maximum_shortening_lengths_s']
        concentric=np.maximum(0,1+np.minimum(shortening,0)/vmax)/(1-np.minimum(shortening,0)/(.25*vmax))
        eccentric=1+.5*np.maximum(shortening,0)/(np.maximum(shortening,0)+.25*vmax)
        fv=np.where(shortening<0,concentric,eccentric)
        self.fiber_force=p['maximum_force_native']*self.activation*fl*fv
        self.last_torque=p['moment_arm_mm']*(self.fiber_force[:,0]-self.fiber_force[:,1])
        self.work_native+=float(self.last_torque@velocity)*dt;self.tick+=1
        return self.last_torque.copy()

    def observe(self,weighted_spikes):
        spikes=np.asarray(weighted_spikes,dtype=float)
        if spikes.shape!=self.pending.shape or not np.isfinite(spikes).all() or (spikes<0).any():
            raise ValueError('Finite nonnegative NMJ events required')
        self.pending+=spikes

    def state(self):
        return {k:getattr(self,k).copy() if isinstance(getattr(self,k),np.ndarray) else getattr(self,k)
                for k in ('activation','rate','pending','saturated','fiber_force','last_torque','work_native','tick')}

    def restore(self,state):
        for name,value in state.items():
            if name not in self.state():raise ValueError('Unknown muscle state')
            current=getattr(self,name)
            if isinstance(current,np.ndarray):
                if not isinstance(value,np.ndarray) or value.shape!=current.shape:raise ValueError('Muscle state shape mismatch')
                current[:]=value
            else:setattr(self,name,value)
