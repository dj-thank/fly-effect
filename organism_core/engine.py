from pathlib import Path
import hashlib
import importlib.metadata as md
import json
import numpy as np
import brian2 as b
from .brain import Brain,ROOT
from .config import SOURCE_ROOT
from .body import Body
from .checkpoint import save,load

SENSORY_IDS=[815569,903843,935541,815843,817680,935383,817298,817839,902324,906823,902359,911316,914116,905037,905335,905339,905340,906290,906558,912317]

def code_hash():
    paths=[SOURCE_ROOT/'organism.py',*sorted((SOURCE_ROOT/'organism_core').glob('*.py'))]
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in paths)).hexdigest()

class Engine:
    def __init__(self,seed,settings=None):
        self.settings={'sensory_mode':'six_leg','input_enabled':True,'perturbation':None,'joint_profile':'generic','muscle_model':'antagonist','load_enabled':True}
        if settings:self.settings.update(settings)
        self.body=Body(self.settings['joint_profile'],self.settings['muscle_model']);self.brain=Brain(seed);self.rng=np.random.default_rng(seed)
        self.load_rng=np.random.default_rng(np.random.SeedSequence([seed,0x4c4f4144]))
        self.seed=seed;self.tick=0;self.input_count=0
        self.frame_ticks=[];self.frame_qpos=[];self.frame_contacts=[];self.frame_force=[]
        from .motor_map import MotorMap
        self.motor_map=MotorMap(self.brain.ids,self.brain.motor,self.body.active_joint_names,ROOT/'runs/motor-inference-v2.json')
        self.body.bind_motors(self.motor_map)
        lookup={int(i):n for n,i in enumerate(self.brain.ids)}
        self.sensory=np.array([lookup[i] for i in SENSORY_IDS],dtype=np.int32)
        import pandas as pd
        from .brain import GRAPH
        ann=pd.read_parquet(GRAPH/'neurons.parquet').set_index('bodyId',verify_integrity=True)
        rows=ann.loc[SENSORY_IDS]
        if not (rows.superclass.eq('vnc_sensory') & rows.subclass.eq('chordotonal organ')).all():
            raise ValueError('Sensory identities mismatch')
        names=self.body.joint_names
        joint=next(i for i,n in enumerate(names) if 'lf_trochanterfemur-lf_tibia-pitch' in n)
        self.sensory_dof=int(self.body.m.jnt_dofadr[joint])
        from .proprioception import Proprioceptors,LEGS,PRIORS
        self.proprioceptors=Proprioceptors(self.body)
        from .load_receptors import LoadReceptors,PRIORS as LOAD_PRIORS
        self.load_receptors=LoadReceptors(self.body)
        self.foot_load_cache=self.body.foot_loads()
        self.sensory_ids=np.array(SENSORY_IDS,dtype=np.int64)
        if self.settings['sensory_mode'] in ('six_leg','six_leg_load'):
            self.sensory_ids=self.proprioceptors.ids.copy()
            if self.settings['sensory_mode']=='six_leg_load':self.sensory_ids=np.r_[self.sensory_ids,self.load_receptors.ids]
            self.sensory=np.array([lookup[int(i)] for i in self.sensory_ids],dtype=np.int32)
        elif self.settings['sensory_mode']!='legacy':raise ValueError('Unknown sensory mode')
        self.driver=b.NeuronGroup(len(self.sensory),'flag : integer',threshold='flag>0',reset='',name='organism_sensory_driver')
        self.input_monitor=b.SpikeMonitor(self.driver,name='organism_input_monitor')
        self.link=b.Synapses(self.driver,self.brain.cells,on_pre='v_post += 68.75*mV',name='organism_sensory_input')
        self.link.connect(i=np.arange(len(self.sensory)),j=self.sensory)
        def start():
            perturb=self.settings['perturbation']
            if perturb is not None and self.tick==perturb['tick']:
                dof=self.proprioceptors.dofs[LEGS.index(perturb['leg'])]
                self.body.d.qvel[dof]+=perturb['velocity_kick_rad_s']
            if self.settings['sensory_mode'] in ('six_leg','six_leg_load'):
                flags=self.proprioceptors.sample(self.body,self.rng,self.settings['input_enabled'])
                if self.settings['sensory_mode']=='six_leg_load':
                    loads=self.load_receptors.sample(self.body,self.load_rng,self.settings['input_enabled'] and self.settings['load_enabled'],forces=self.foot_load_cache)
                    flags=np.r_[flags,loads]
            else:
                speed=abs(float(self.body.d.qvel[self.sensory_dof]))
                rate=100*speed/(5+speed)
                flags=self.rng.random(len(self.sensory))<rate*.0001
                if not self.settings['input_enabled']:flags[:]=False
            self.driver.flag=flags.astype(np.int32);self.input_count+=int(flags.sum())
        def end():
            # Muscle commands use only spikes through the previous neural tick.
            self.body.step()
            self.foot_load_cache=self.body.foot_loads()
            self.body.observe_spikes(self.brain.cells.spikes,self.motor_map)
            self.tick+=1
            if abs(self.body.d.time-self.tick*.0001)>1e-10:raise ValueError('Body clock mismatch')
            if self.tick%10==0:
                self.frame_ticks.append(self.tick);self.frame_qpos.append(self.body.d.qpos.copy())
                feet,_=self.body.contacts();self.frame_contacts.append([feet[leg] for leg in LEGS])
                self.frame_force.append(float(np.max(self.body.muscles.fiber_force)))
        self.start=b.NetworkOperation(start,when='start',order=-100,name='organism_observe')
        self.end=b.NetworkOperation(end,when='end',order=100,name='organism_evolve')
        self.brain.network.add(self.driver,self.link,self.start,self.end,self.input_monitor)
        self.identity={'schema':1,'code_sha256':code_hash(),'graph_sha256':self.brain.graph_hash,
                       'body_sha256':self.body.digest,'seed':seed,
                       'settings':self.settings,'proprioception_priors':PRIORS,'load_priors':LOAD_PRIORS,
                       'joint_profile':self.body.joint_profile,'model_source_hashes':self.body.model_source_hashes,
                       'motor_map_source_hashes':self.motor_map.source_hashes,
                       'versions':{k:md.version(k) for k in ('numpy','brian2','mujoco','flygym')}}

    def run(self,seconds):
        ticks=round(seconds/.0001)
        if ticks<1 or abs(seconds-ticks*.0001)>1e-12:raise ValueError('Integer 100-us duration required')
        self.brain.run(seconds)
        assert self.tick==round(float(self.brain.network.t/b.second)/.0001)

    def snapshot(self,path):
        return save(path,{'neural':self.brain.state(),'body':self.body.state(),
                          'muscles':self.body.muscles.state(),'motor_map':self.motor_map.state(),
                          'proprioceptors':self.proprioceptors.state(),'load_receptors':self.load_receptors.state(),'load_rng':self.load_rng.bit_generator.state,'foot_load_cache':self.foot_load_cache.copy(),
                          'frames':{'ticks':np.asarray(self.frame_ticks,dtype=np.int64),'qpos':np.asarray(self.frame_qpos),
                                    'contacts':np.asarray(self.frame_contacts,dtype=bool),'force':np.asarray(self.frame_force)},
                          'rng':self.rng.bit_generator.state,'tick':self.tick,'input_count':self.input_count},self.identity)

    def restore(self,path,digest):
        state=load(path,self.identity,digest)
        self.brain.restore(state['neural']);self.body.restore(state['body'])
        self.body.muscles.restore(state['muscles']);self.motor_map.restore(state['motor_map'])
        self.proprioceptors.restore(state['proprioceptors']);self.load_receptors.restore(state['load_receptors'])
        self.load_rng.bit_generator.state=state['load_rng']
        self.foot_load_cache=state['foot_load_cache'].copy()
        self.frame_ticks=state['frames']['ticks'].tolist();self.frame_qpos=list(state['frames']['qpos'])
        self.frame_contacts=state['frames']['contacts'].tolist();self.frame_force=state['frames']['force'].tolist()
        self.rng.bit_generator.state=state['rng'];self.tick=state['tick'];self.input_count=state['input_count']

    def observation(self):
        result=self.brain.observation();result['body']=self.body.state()
        result['sensory_i']=np.asarray(self.input_monitor.i[:]).copy()
        result['sensory_t']=np.asarray(self.input_monitor.t[:]/b.second).copy()
        result['sensory_body_ids']=self.sensory_ids.copy()
        result['frame_ticks']=np.asarray(self.frame_ticks,dtype=np.int64)
        result['frame_qpos']=np.asarray(self.frame_qpos)
        result['frame_contacts']=np.asarray(self.frame_contacts,dtype=bool)
        result['frame_muscle_force']=np.asarray(self.frame_force)
        result['foot_load_cache']=self.foot_load_cache.copy()
        result['load_filtered']=self.load_receptors.filtered.copy();result['load_last_force']=self.load_receptors.last_force.copy()
        result['load_rates']=self.load_receptors.last_rate.copy();result['load_counts']=self.load_receptors.counts.copy()
        result['receptor_adaptation']=self.proprioceptors.adaptation.copy()
        result['muscle_activation']=self.body.muscles.activation.copy()
        result['muscle_rate']=self.body.muscles.rate.copy()
        result['muscle_pending']=self.body.muscles.pending.copy()
        result['muscle_force']=self.body.muscles.fiber_force.copy()
        result['muscle_torque']=self.body.muscles.last_torque.copy()
        result['rng']=self.rng.bit_generator.state;result['inputs']=self.input_count
        return result

    def report(self):
        observation=self.observation()
        hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in observation.items() if isinstance(v,np.ndarray)}
        counts=np.bincount(observation['spike_i'],minlength=len(self.brain.ids))
        from .sensorimotor_coverage import summarize
        coverage=summarize(self.brain.ids,self.brain.motor,self.motor_map.rows,
                           self.motor_map.assignments,self.sensory_ids,self.motor_map.state(),
                           joint_names=self.body.active_joint_names,
                           motor_spikes=int(counts[self.brain.motor].sum()))
        coverage.update(sensory_mode=self.settings['sensory_mode'],body_input_enabled=self.settings['input_enabled'])
        return {'identity':self.identity,'ticks':self.tick,'input_count':self.input_count,
                'spikes':len(observation['spike_i']),'motor_spikes':int(counts[self.brain.motor].sum()),
                'observation_hashes':hashes,'rng':{'proprioceptive':self.rng.bit_generator.state,'load':self.load_rng.bit_generator.state},
                'load_input_count':int(self.load_receptors.counts.sum()),
                'neuron_count':len(self.brain.ids),'edge_count':len(self.brain.edges),
                'free_body':True,'body_actuators':self.body.m.nu,
                'external_muscle_units':int(self.body.muscles.activation.size),
                'motor_coupling_implemented':True,'sensorimotor_coverage':coverage,'motor_coverage':{'total':len(self.motor_map.rows),'required':sum(r['required_for_walking'] for r in self.motor_map.rows),
                    'mapped':len(self.motor_map.assignments),'unresolved_required':sum(r['required_for_walking'] and r['joint'] is None for r in self.motor_map.rows)},
                'tendon_coverage':({'motor_ids':len(self.body.muscles.migrated),'uninnervated_tendons':self.body.muscles.tendons.uninnervated,'motor_events':self.body.muscles.tendons.events} if self.settings['muscle_model']=='tendon_candidate' else None),
                'motor_accounting':self.motor_map.state(),'walking_passed':False,
                'sensory_rule_status':'ASSUMED class-dependent position/direction/movement response; not calibrated',
                'sensory_count':len(self.sensory),'sensory_mode':self.settings['sensory_mode'],
                'peak_muscle_force_native':max(self.frame_force,default=0.),
                'functional_gates_automatically_passed':[]}
