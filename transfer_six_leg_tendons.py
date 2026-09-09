"""Explicit six-leg extrapolation of the validated LF registration construction."""
from pathlib import Path
from xml.etree import ElementTree as ET
import hashlib,json
import numpy as np
import mujoco as mj
from organism_core.body import Body
from organism_core.joint_geometry import neutral_geometry
from transfer_lf_tendons import body_id,frame,pulling_force

from organism_core.config import HOME as ROOT, DATA
LEGS=('lf','lm','lh','rf','rm','rh')

def main():
    source=ROOT/'runs/source-tendon-audit/kinematic-only.xml';audit=json.loads((source.parent/'results.json').read_text(encoding='utf-8'))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=audit['derived_xml_sha256']:raise ValueError('Source changed')
    sm=mj.MjModel.from_xml_path(str(source));sd=mj.MjData(sm)
    mj.mj_resetDataKeyframe(sm,sd,mj.mj_name2id(sm,mj.mjtObj.mjOBJ_KEY,'default-pose'));mj.mj_forward(sm,sd)
    body=Body('muscle_compliance');tm,td=body.m,body.d;mj.mj_forward(tm,td)
    def point(m,d,name):return d.xpos[body_id(m,name)].copy()
    source_bones={'Thorax':['LFCoxa','RFCoxa','LMCoxa'],'LFCoxa':['LFCoxa','LFTrochanter','LFTibia'],
                  'combined':['LFTrochanter','LFTibia','LFTarsus1'],'LFTibia':['LFTibia','LFTarsus1','LFTrochanter']}
    source_frames={key:frame(np.array([point(sm,sd,n) for n in names])) for key,names in source_bones.items()}
    source_xml=ET.fromstring(source.read_text(encoding='utf-8'));xml=ET.fromstring(body.xml)
    bodies={e.get('name'):e for e in xml.iter('body')};tendon_block=ET.SubElement(xml,'tendon');sites=[];transform_rows=[]
    for leg in LEGS:
        other=('r' if leg[0]=='l' else 'l')+leg[1]
        name=lambda part:f'organism/{leg}_{part}'
        origin=point(tm,td,name('coxa'));opposite=point(tm,td,f'organism/{other}_coxa')
        if leg[1]!='h':caudal=point(tm,td,f"organism/{leg[0]}{'m' if leg[1]=='f' else 'h'}_coxa")
        else:caudal=origin+(origin-point(tm,td,f'organism/{leg[0]}m_coxa'))
        target_points={'Thorax':np.array([origin,opposite,caudal]),
                       'LFCoxa':np.array([origin,point(tm,td,name('trochanterfemur')),point(tm,td,name('tibia'))]),
                       'combined':np.array([point(tm,td,name('trochanterfemur')),point(tm,td,name('tibia')),point(tm,td,name('tarsus1'))]),
                       'LFTibia':np.array([point(tm,td,name('tibia')),point(tm,td,name('tarsus1')),point(tm,td,name('trochanterfemur'))])}
        transforms={}
        for key,points in target_points.items():
            so,sr,sl=source_frames[key];to,tr,tl=frame(points)
            if leg[0]=='r':tr[:,1]*=-1 # Reflect chirality, not merely rotate a left leg.
            transforms[key]=(so,sr,to,tr,tl/sl)
            transform_rows.append({'leg':leg,'bone':key,'scale':tl/sl,'determinant':float(np.linalg.det(tr@sr.T))})
        destination={'Thorax':'organism/c_thorax','LFCoxa':name('coxa'),'LFTrochanter':name('trochanterfemur'),
                     'LFFemur':name('trochanterfemur'),'LFTibia':name('tibia')}
        for element in source_xml.findall('./tendon/spatial'):
            transferred=ET.SubElement(tendon_block,'spatial',name=f'candidate_{leg}_'+element.get('name'),width='.005')
            for ref in element:
                if ref.tag!='site':raise ValueError('Unsupported wrap primitive')
                source_name=ref.get('site');sid=mj.mj_name2id(sm,mj.mjtObj.mjOBJ_SITE,source_name)
                parent=mj.mj_id2name(sm,mj.mjtObj.mjOBJ_BODY,int(sm.site_bodyid[sid]));target=destination[parent]
                key='combined' if parent in ('LFTrochanter','LFFemur') else parent
                so,sr,to,tr,scale=transforms[key];world=to+tr@((sr.T@(sd.site_xpos[sid]-so))*scale)
                tid=body_id(tm,target);local=td.xmat[tid].reshape(3,3).T@(world-td.xpos[tid])
                new_name=f'candidate_{leg}_'+source_name
                if not any(e.get('name')==new_name for e in bodies[target].findall('site')):
                    ET.SubElement(bodies[target],'site',name=new_name,pos=' '.join(format(v,'.17g') for v in local),size='.007')
                    sites.append({'leg':leg,'source_site':source_name,'target_body':target,'target_local':local.tolist(),'bone':key})
                ET.SubElement(transferred,'site',site=new_name)
    candidate_xml=ET.tostring(xml,encoding='unicode');model=mj.MjModel.from_xml_string(candidate_xml);data=mj.MjData(model)
    mj.mj_resetDataKeyframe(model,data,mj.mj_name2id(model,mj.mjtObj.mjOBJ_KEY,'neutral'));mj.mj_forward(model,data)
    geometry=neutral_geometry(model);neutral=data.qpos.copy()
    tendons=[[mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,e.get('site')) for e in t] for t in tendon_block]
    rows=[];max_error=0.;max_base=0.;directions={}
    for index,t in enumerate(tendon_block):
        data.qpos[:]=neutral;mj.mj_forward(model,data);tensions=np.zeros(model.ntendon);tensions[index]=1.
        tau=pulling_force(model,data,tendons,tensions);max_base=max(max_base,float(np.max(np.abs(tau[:6]))))
        finite=np.zeros(model.nv)
        for j in body.active_joint_ids:
            q=model.jnt_qposadr[j];d=model.jnt_dofadr[j];eps=1e-6
            data.qpos[:]=neutral;data.qpos[q]+=eps;mj.mj_forward(model,data);plus=data.ten_length[index]
            data.qpos[:]=neutral;data.qpos[q]-=eps;mj.mj_forward(model,data);minus=data.ten_length[index]
            finite[d]=-(plus-minus)/(2*eps)
        error=float(np.max(np.abs(tau-finite)));max_error=max(max_error,error)
        name=t.get('name');leg=name.split('_')[1]
        if 'LFTibia_flex' in name or 'LFTibia_extensor' in name:
            g=geometry[leg+':knee'];signed=tau[model.jnt_dofadr[g['joint_id']]]*g['extension_sign']
            directions[name]=bool(signed<0 if 'LFTibia_flex' in name else signed>0)
        rows.append({'name':name,'leg':leg,'unit_tension_torque':{body.joint_names[j]:float(tau[model.jnt_dofadr[j]]) for j in body.active_joint_ids if abs(tau[model.jnt_dofadr[j]])>1e-8}})
    if max_error>1e-7 or max_base>1e-10 or len(directions)!=12 or not all(directions.values()):raise ValueError('Geometry check failed')
    out=ROOT/'runs/six-leg-tendon-candidate';out.mkdir(exist_ok=True);(out/'body.xml').write_text(candidate_xml,encoding='utf-8',newline='\n')
    result={'source_sha256':audit['source_sha256'],'base_body_sha256':body.digest,'candidate_xml_sha256':hashlib.sha256(candidate_xml.encode()).hexdigest(),
            'transforms':transform_rows,'sites':sites,'tendons':rows,'knee_direction_checks':directions,'max_virtual_work_error':max_error,'max_free_base_force':max_base,
            'assumption':'LF geometry extrapolated to all legs with bone frames, uniform scaling and reflection for right legs; hind thorax caudal landmark extrapolated from ipsilateral middle-to-hind displacement.',
            'status':'CALIBRATION_CANDIDATE','biological_validation':False,'CNS_executed':False,'runtime_integrated':False,'walking_passed':False}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('transforms','sites','tendons')},indent=2))

if __name__=='__main__':main()
