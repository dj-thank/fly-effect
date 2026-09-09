"""Determine flexion/extension coordinates from actual adjacent joint anchors."""
import numpy as np
import mujoco as mj

LEGS=('lf','lm','lh','rf','rm','rh')

def internal_angle(data,proximal,pivot,distal):
    a=data.xanchor[proximal]-data.xanchor[pivot]
    b=data.xanchor[distal]-data.xanchor[pivot]
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    if norm<=0:raise ValueError('Degenerate joint geometry')
    return float(np.arccos(np.clip(a@b/norm,-1,1)))

def neutral_geometry(model):
    data=mj.MjData(model)
    mj.mj_resetDataKeyframe(model,data,mj.mj_name2id(model,mj.mjtObj.mjOBJ_KEY,'neutral'))
    neutral=data.qpos.copy()
    names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt)]
    result={}
    for leg in LEGS:
        chain=[next(i for i,n in enumerate(names) if n.endswith(f'-{leg}_{segment}-pitch')) for segment in ('coxa','trochanterfemur','tibia','tarsus1')]
        for kind,triple in (('hip',chain[:3]),('knee',chain[1:])):
            joint=triple[1];qpos=int(model.jnt_qposadr[joint])
            angles=[]
            for delta in (0.,.001):
                data.qpos[:]=neutral;data.qpos[qpos]+=delta;mj.mj_forward(model,data)
                angles.append(internal_angle(data,*triple))
            derivative=(angles[1]-angles[0])/.001
            if not .9<abs(derivative)<1.1:raise ValueError('Joint angle not locally represented by a single hinge')
            result[leg+':'+kind]={'joint_name':names[joint],'joint_id':joint,'qpos':qpos,
                'native_neutral':float(neutral[qpos]),'angle_neutral':angles[0],
                'extension_sign':1 if derivative>0 else -1,'derivative':derivative,
                'anchor_joint_ids':triple}
    return result
