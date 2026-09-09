"""Extract force moment arms from pinned FlyMimic tendon geometry, not muscle names."""
from pathlib import Path
from xml.etree import ElementTree as ET
import hashlib
import json
import numpy as np
import mujoco as mj

from organism_core.config import HOME as ROOT, DATA

def main():
    from flygym.compose.fly.musculoskeletal import DEFAULT_MUSCULOSKELETAL_XML
    source=DEFAULT_MUSCULOSKELETAL_XML;raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!='04f6070d6733940357be005ca72c02ba0d9455538ff018da70c74de7458e9531':raise ValueError('Unexpected source')
    tree=ET.fromstring(raw)
    parameters={e.get('tendon'):{'muscle':e.get('name'),'source_gainprm':list(map(float,e.get('gainprm').split()))} for e in tree.findall('./actuator/general') if e.get('class')=='muscle'}
    # Remove collision/rendering and activation only for a no-dynamics geometry audit.
    for parent in list(tree.iter()):
        for child in list(parent):
            if child.tag in ('asset','geom','contact','sensor','actuator','equality','camera','light'):parent.remove(child)
    xml=ET.tostring(tree,encoding='unicode');model=mj.MjModel.from_xml_string(xml);data=mj.MjData(model)
    mj.mj_resetDataKeyframe(model,data,mj.mj_name2id(model,mj.mjtObj.mjOBJ_KEY,'default-pose'))
    q0=data.qpos.copy();mj.mj_forward(model,data);lengths=data.ten_length.copy()
    joint_names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,j) for j in range(model.njnt)]
    estimates=[]
    for eps in (1e-5,5e-6):
        jac=np.zeros((model.ntendon,model.njnt))
        for j in range(model.njnt):
            pos=model.jnt_qposadr[j]
            data.qpos[:]=q0;data.qpos[pos]+=eps;mj.mj_forward(model,data);plus=data.ten_length.copy()
            data.qpos[:]=q0;data.qpos[pos]-=eps;mj.mj_forward(model,data);minus=data.ten_length.copy()
            jac[:,j]=(plus-minus)/(2*eps)
        estimates.append(jac)
    error=float(np.max(np.abs(estimates[0]-estimates[1])))
    if error>1e-7:raise ValueError('Unstable finite difference')
    rows=[]
    for t in range(model.ntendon):
        tendon=mj.mj_id2name(model,mj.mjtObj.mjOBJ_TENDON,t)
        if tendon not in parameters:continue
        # A tensile force pulls to decrease tendon length: tau = -dL/dq * F.
        moments=-estimates[1][t]
        rows.append({**parameters[tendon],'tendon':tendon,'length_native':float(lengths[t]),
            'unit_tension_torque_by_joint':{n:float(v) for n,v in zip(joint_names,moments) if abs(v)>1e-8},
            'dominant_joint':joint_names[int(np.argmax(np.abs(moments)))]})
    out=ROOT/'runs/source-tendon-audit';out.mkdir(exist_ok=True)
    (out/'kinematic-only.xml').write_text(xml,encoding='utf-8',newline='\n')
    result={'source_path':str(source),'source_sha256':digest,'derived_xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),
        'pose':'source default-pose keyframe','finite_difference_error':error,'muscles':rows,
        'scope':'Fixed LF source kinematics only. Native source joint axes are not assumed identical to composed six-leg axes.',
        'CNS_executed':False,'walking_passed':False,'source_to_six_leg_registration_verified':False}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for r in rows:print(r['muscle'],r['unit_tension_torque_by_joint'])
    print('Finite difference error:',error)

if __name__=='__main__':main()
