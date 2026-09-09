"""Portable CPU LIF backend based on published Shiu et al. equations.

Independent wiring adapter; does not import or redistribute Eon's GPL backend.
See THIRD_PARTY_NOTICES.md for the MIT reference and scientific attribution.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import brian2 as b
from .config import HOME as ROOT, GRAPH

EDGE_DTYPE=np.dtype([('pre','<u4'),('post','<u4'),('count','<u4'),('source_row','<u8')])

def load_graph(directory):
    directory=Path(directory)
    lock=json.loads(Path(__file__).with_name('graph_lock.json').read_text(encoding='utf-8'))
    for name,expected in lock['files'].items():
        path=directory/name
        if not path.is_file():raise FileNotFoundError(f'Missing graph artifact: {name}. See docs/DATA.md.')
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:raise ValueError('Graph integrity mismatch: '+name)
    ids=np.load(directory/'body_ids.npy',allow_pickle=False)
    motor=np.load(directory/'motor_indices.npy',allow_pickle=False)
    signs=np.load(directory/'glutamate_inhibitory_hypothesis_signs.npy',allow_pickle=False)
    edges=np.memmap(directory/'edges.bin',dtype=EDGE_DTYPE,mode='r')
    if len(ids)!=lock['neurons'] or len(edges)!=lock['edges'] or len(set(ids))!=len(ids):raise ValueError('Graph dimensions or identity mismatch')
    if signs.shape!=ids.shape or not np.isin(signs,[-1,0,1]).all():raise ValueError('Invalid sign hypothesis')
    if edges['pre'].max()>=len(ids) or edges['post'].max()>=len(ids):raise ValueError('Invalid edge index')
    if len(motor)!=815 or motor.min()<0 or motor.max()>=len(ids) or len(set(motor))!=815:raise ValueError('Invalid motor index set')
    return ids,motor,signs,edges,lock['files']['edges.bin']

class Brain:
    def __init__(self,seed=0):
        self.ids,self.motor,self.signs,self.edges,self.graph_hash=load_graph(GRAPH)
        self._construct(len(self.ids),self.edges['pre'].astype(np.int32),self.edges['post'].astype(np.int32),self.edges['count'].astype(np.float64)*self.signs[self.edges['pre']],seed)

    @classmethod
    def synthetic(cls,seed=0):
        obj=cls.__new__(cls)
        obj.ids=np.array([1,2,3]);obj.motor=np.array([2]);obj.signs=np.array([1,1,0])
        obj.edges=np.array([(0,1,300,0),(1,2,300,1)],dtype=EDGE_DTYPE)
        obj.graph_hash='synthetic-example-not-biological-data'
        obj._construct(3,np.array([0,1]),np.array([1,2]),np.array([300.,300.]),seed)
        return obj

    def _construct(self,n,pre,post,weights,seed):
        b.start_scope();b.seed(seed);b.defaultclock.dt=.1*b.ms;b.prefs.codegen.target='numpy'
        self.params={'v_0':-52*b.mV,'v_rst':-52*b.mV,'v_th':-45*b.mV,'t_mbr':20*b.ms,'tau':5*b.ms,'t_rfc':2.2*b.ms,'t_dly':1.8*b.ms,'w_syn':.275*b.mV}
        self.cells=b.NeuronGroup(n,'dv/dt = (v_0-v+g)/t_mbr : volt (unless refractory)\ndg/dt = -g/tau : volt (unless refractory)\nrfc : second',threshold='v>v_th',reset='v=v_rst; g=0*mV',refractory='rfc',method='linear',namespace=self.params,name='default_neurons')
        self.cells.v=self.params['v_0'];self.cells.g=0*b.mV;self.cells.rfc=self.params['t_rfc']
        self.syn=b.Synapses(self.cells,self.cells,'w : volt',on_pre='g_post += w',delay=self.params['t_dly'],name='default_synapses')
        self.syn.connect(i=pre,j=post);self.syn.w=weights*self.params['w_syn']
        self.spikes=b.SpikeMonitor(self.cells,name='default_spike_monitor')
        self.network=b.Network(self.cells,self.syn,self.spikes)

    def run(self,seconds):self.network.run(seconds*b.second,namespace={})
    def state(self):
        self.network.store('portable');return self.network._stored_state['portable']
    def restore(self,state):
        self.network._stored_state['portable']=state;self.network.restore('portable',restore_random_state=True)
    def observation(self):
        return {'tick':round(float(self.network.t/b.second)/.0001),'v_mV':np.asarray(self.cells.v[:]/b.mV).copy(),'g_mV':np.asarray(self.cells.g[:]/b.mV).copy(),'spike_i':np.asarray(self.spikes.i[:]).copy(),'spike_t':np.asarray(self.spikes.t[:]/b.second).copy()}
