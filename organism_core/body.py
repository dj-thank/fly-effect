"""Existing NeuroMechFly free six-leg plant, with no movement controller."""
from .config import HOME
from .contacts import is_active_contact
import hashlib
import numpy as np
import mujoco as mj
from pathlib import Path
import json
from xml.etree import ElementTree as ET

UNIT_CONTRACT={'length':'mm','time':'s','mass':'g','force':'g*mm/s^2','torque':'g*mm^2/s^2',
               'force_to_N':1e-6,'torque_to_Nm':1e-9,
               'mass_interpretation':'ASSUMED gram convention applied to transferred native mass values, not a new biological measurement',
               'joint_half_range_rad':float(np.pi/2),'joint_limits_status':'ASSUMED ±90deg, intersected with linearized hip/knee internal angles [0.1, pi-0.1]'}


class Body:
    def __init__(self,joint_profile='generic',muscle_model='antagonist'):
        from flygym.compose import NeuroMechFly,FlatGroundWorld,KinematicPosePreset
        from flygym.anatomy import Skeleton,JointPreset,AxisOrder
        from flygym.utils.math import Rotation3D
        if muscle_model not in ('antagonist','tendon_candidate'):raise ValueError('Unknown muscle model')
        self.muscle_model=muscle_model;self.model_source_hashes={}
        self.fly=NeuroMechFly(name='organism')
        self.joint_profile={'name':joint_profile,'status':'ASSUMED/TRANSFERRED, not calibrated',
                            'stiffness':10.,'damping':.5,'armature':1e-6}
        if joint_profile!='generic':
            if joint_profile not in ('muscle_compliance','muscle_transfer'):raise ValueError('Unknown joint profile')
            from flygym.compose.fly.musculoskeletal import DEFAULT_MUSCULOSKELETAL_XML
            raw=DEFAULT_MUSCULOSKELETAL_XML.read_bytes()
            digest=hashlib.sha256(raw).hexdigest()
            if digest!='04f6070d6733940357be005ca72c02ba0d9455538ff018da70c74de7458e9531':raise ValueError('Muscle prior source changed')
            prior=ET.fromstring(raw).find('./default/joint')
            self.joint_profile.update(stiffness=float(prior.get('stiffness')),damping=float(prior.get('damping')),
                                      source_sha256=digest,source_scope='fixed LF muscle model transferred to active six-leg joints')
            if joint_profile=='muscle_transfer':self.joint_profile['armature']=float(prior.get('armature'))
        skeleton=Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL,joint_preset=JointPreset.LEGS_ONLY)
        self.fly.add_joints(skeleton,neutral_pose=KinematicPosePreset.NEUTRAL)
        world=FlatGroundWorld(half_size=20)
        world.add_fly(self.fly,spawn_position=(0,0,0.1),spawn_rotation=Rotation3D('quat',(1,0,0,0)))
        # MjSpec can emit inferred STL content_type on different mirrored meshes.
        # Make the inferred file format explicit, then compile exactly this XML.
        tree=ET.fromstring(world.mjcf_root.to_xml())
        asset_hashes={}
        self.asset_files={}
        for mesh in tree.findall('./asset/mesh'):
            path=Path(mesh.attrib['file'])
            if path.suffix.lower()!='.stl':raise ValueError('Unexpected mesh format')
            mesh.set('content_type','model/stl')
            asset_hashes[mesh.attrib['name']]=hashlib.sha256(path.read_bytes()).hexdigest()
            self.asset_files[str(path)]=asset_hashes[mesh.attrib['name']]
        for element in tree.iter():
            if element.tag=='joint':
                neutral=float(element.get('springref','0'))
                half=UNIT_CONTRACT['joint_half_range_rad']
                element.set('limited','true');element.set('range',f'{neutral-half:.17g} {neutral+half:.17g}')
                if joint_profile!='generic' and not any(f'tarsus{k}' in element.get('name','') for k in range(2,6)):
                    for key in ('stiffness','damping','armature'):element.set(key,format(self.joint_profile[key],'.17g'))
            attributes=sorted(element.attrib.items())
            element.attrib.clear();element.attrib.update(attributes)
        self.xml=ET.tostring(tree,encoding='unicode')
        provisional=mj.MjModel.from_xml_string(self.xml)
        from .joint_geometry import neutral_geometry
        self.joint_geometry=neutral_geometry(provisional)
        elements={e.get('name'):e for e in tree.iter('joint')}
        for geometry in self.joint_geometry.values():
            q0=geometry['native_neutral'];a0=geometry['angle_neutral'];sign=geometry['extension_sign']
            bounds=sorted([q0+(.1-a0)/sign,q0+(np.pi-.1-a0)/sign])
            old=provisional.jnt_range[geometry['joint_id']]
            bounds=[max(float(old[0]),bounds[0]),min(float(old[1]),bounds[1])]
            if not bounds[0]<q0<bounds[1]:raise ValueError('Neutral outside anatomical angle limits')
            elements[geometry['joint_name']].set('range',' '.join(format(x,'.17g') for x in bounds))
        self.xml=ET.tostring(tree,encoding='unicode')
        if muscle_model=='tendon_candidate':
            if joint_profile!='muscle_compliance':raise ValueError('Tendon candidate requires muscle_compliance profile')
            from .config import HOME
            root=HOME
            directory=root/'runs/six-leg-tendon-candidate'
            receipt=json.loads((directory/'results.json').read_text(encoding='utf-8'))
            base=hashlib.sha256(self.xml.encode()+json.dumps(asset_hashes,sort_keys=True).encode()).hexdigest()
            if base!=receipt['base_body_sha256']:raise ValueError('Tendon candidate base changed')
            raw=(directory/'body.xml').read_bytes()
            if hashlib.sha256(raw).hexdigest()!=receipt['candidate_xml_sha256']:raise ValueError('Tendon geometry changed')
            self.tendon_source_audit=root/'runs/source-tendon-audit/results.json'
            audit=json.loads(self.tendon_source_audit.read_text(encoding='utf-8'))
            source=Path(audit['source_path'])
            if hashlib.sha256(source.read_bytes()).hexdigest()!=receipt['source_sha256']:raise ValueError('Tendon source changed')
            files=[directory/'body.xml',directory/'results.json',self.tendon_source_audit,source]
            self.model_source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            self.xml=raw.decode('utf-8')
        self.m=mj.MjModel.from_xml_string(self.xml)
        self.d=mj.MjData(self.m)
        neutral=mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_KEY,'neutral')
        mj.mj_resetDataKeyframe(self.m,self.d,neutral)
        self.joint_names=[mj.mj_id2name(self.m,mj.mjtObj.mjOBJ_JOINT,i) for i in range(self.m.njnt)]
        self.active_joint_ids=np.array([i for i,n in enumerate(self.joint_names) if i and not any(f'tarsus{k}' in n for k in range(2,6))],dtype=int)
        self.active_joint_names=[self.joint_names[i] for i in self.active_joint_ids]
        self.active_qpos=self.m.jnt_qposadr[self.active_joint_ids]
        self.active_dofs=self.m.jnt_dofadr[self.active_joint_ids]
        from .muscles import MuscleBank
        self.muscles=MuscleBank(self.d.qpos[self.active_qpos])
        self.digest=hashlib.sha256(self.xml.encode()+json.dumps(asset_hashes,sort_keys=True).encode()).hexdigest()
        assert np.count_nonzero(self.m.jnt_type==mj.mjtJoint.mjJNT_FREE)==1

    def bind_motors(self,mapping):
        self.muscles.pool_sizes=mapping.pool_sizes.copy()
        if self.muscle_model=='tendon_candidate':
            from .tendon_coupling import HybridMuscles
            self.muscles=HybridMuscles(self,mapping,self.tendon_source_audit)

    def observe_spikes(self,indices,mapping):
        if self.muscle_model=='tendon_candidate':self.muscles.observe_neurons(indices,mapping)
        else:self.muscles.observe(mapping.decode(indices))

    def step(self):
        torque=self.muscles.command(self.d.qpos[self.active_qpos],self.d.qvel[self.active_dofs])
        self.d.qfrc_applied[:]=0
        self.d.qfrc_applied[self.active_dofs]=torque
        mj.mj_step(self.m,self.d)
        if np.any(self.d.warning.number) or not np.isfinite(np.r_[self.d.qpos,self.d.qvel,self.d.qacc]).all():
            raise ValueError('Nonfinite body or solver warning')

    def state(self):
        specification=mj.mjtState.mjSTATE_INTEGRATION
        values=np.empty(mj.mj_stateSize(self.m,specification))
        mj.mj_getState(self.m,self.d,values,specification)
        return values

    def restore(self,values):
        mj.mj_setState(self.m,self.d,values,mj.mjtState.mjSTATE_INTEGRATION)
        mj.mj_forward(self.m,self.d)

    def contacts(self):
        feet={leg:False for leg in ('lf','lm','lh','rf','rm','rh')}
        reaction=np.zeros(3)
        for i in range(self.d.ncon):
            contact=self.d.contact[i]
            if not is_active_contact(contact):continue
            names=[mj.mj_id2name(self.m,mj.mjtObj.mjOBJ_GEOM,g) or '' for g in (contact.geom1,contact.geom2)]
            if 'ground_plane' not in names:continue
            ground_first=names[0]=='ground_plane';other=names[1 if ground_first else 0]
            for leg in feet:
                if f'/{leg}_tarsus' in other:feet[leg]=True
            force=np.zeros(6);mj.mj_contactForce(self.m,self.d,i,force)
            reaction+=(1 if ground_first else -1)*(contact.frame.reshape(3,3).T@force[:3])
        return feet,reaction

    def foot_loads(self):
        legs=('lf','lm','lh','rf','rm','rh');forces=np.zeros((6,3))
        for i in range(self.d.ncon):
            contact=self.d.contact[i]
            if not is_active_contact(contact):continue
            names=[mj.mj_id2name(self.m,mj.mjtObj.mjOBJ_GEOM,int(g)) or '' for g in (contact.geom1,contact.geom2)]
            if 'ground_plane' not in names:continue
            first=names[0]=='ground_plane';other=names[1 if first else 0]
            matches=[j for j,leg in enumerate(legs) if f'/{leg}_tarsus' in other]
            if len(matches)!=1:continue
            local=np.zeros(6);mj.mj_contactForce(self.m,self.d,i,local)
            forces[matches[0]]+=(1 if first else -1)*(contact.frame.reshape(3,3).T@local[:3])
        return forces
