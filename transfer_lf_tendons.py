"""Register source LF tendon sites to composed LF bones as an explicit candidate."""
from pathlib import Path
from xml.etree import ElementTree as ET
import copy
import hashlib
import json
import numpy as np
import mujoco as mj
from organism_core.body import Body

from organism_core.config import HOME as ROOT, DATA

def body_id(m,name):
    result=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,name)
    if result<0:raise ValueError('Missing body '+name)
    return result

def frame(points):
    origin,end,plane=points
    z=end-origin;length=np.linalg.norm(z);z=z/length
    x=plane-origin;x=x-z*(x@z);x=x/np.linalg.norm(x)
    return origin,np.column_stack((x,np.cross(z,x),z)),length

def pulling_force(model,data,tendons,tensions):
    generalized=np.zeros(model.nv)
    for sites,tension in zip(tendons,tensions):
        if tension==0:continue
        for a,b in zip(sites[:-1],sites[1:]):
            delta=data.site_xpos[b]-data.site_xpos[a];length=np.linalg.norm(delta)
            if length<=1e-12:raise ValueError('Degenerate tendon segment')
            force=float(tension)*delta/length
            for site,value in ((a,force),(b,-force)):
                mj.mj_applyFT(model,data,value,np.zeros(3),data.site_xpos[site],int(model.site_bodyid[site]),generalized)
    return generalized

def main():
    source_path=ROOT/'runs/source-tendon-audit/kinematic-only.xml'
    audit=json.loads((source_path.parent/'results.json').read_text(encoding='utf-8'))
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=audit['derived_xml_sha256']:raise ValueError('Changed source geometry')
    sm=mj.MjModel.from_xml_path(str(source_path));sd=mj.MjData(sm)
    mj.mj_resetDataKeyframe(sm,sd,mj.mj_name2id(sm,mj.mjtObj.mjOBJ_KEY,'default-pose'));mj.mj_forward(sm,sd)
    body=Body('muscle_compliance');tm,td=body.m,body.d;mj.mj_forward(tm,td)
    def points(m,d,names):return np.array([d.xpos[body_id(m,n)] for n in names])
    source_bones={'Thorax':['LFCoxa','RFCoxa','LMCoxa'],'LFCoxa':['LFCoxa','LFTrochanter','LFTibia'],
                  'combined':['LFTrochanter','LFTibia','LFTarsus1'],'LFTibia':['LFTibia','LFTarsus1','LFTrochanter']}
    target_bones={'Thorax':['organism/lf_coxa','organism/rf_coxa','organism/lm_coxa'],
                  'LFCoxa':['organism/lf_coxa','organism/lf_trochanterfemur','organism/lf_tibia'],
                  'combined':['organism/lf_trochanterfemur','organism/lf_tibia','organism/lf_tarsus1'],
                  'LFTibia':['organism/lf_tibia','organism/lf_tarsus1','organism/lf_trochanterfemur']}
    transforms={}
    for key in source_bones:
        so,sr,sl=frame(points(sm,sd,source_bones[key]));to,tr,tl=frame(points(tm,td,target_bones[key]))
        transforms[key]=(so,sr,to,tr,tl/sl)
    destination={'Thorax':'organism/c_thorax','LFCoxa':'organism/lf_coxa','LFTrochanter':'organism/lf_trochanterfemur',
                 'LFFemur':'organism/lf_trochanterfemur','LFTibia':'organism/lf_tibia'}
    xml=ET.fromstring(body.xml);bodies={e.get('name'):e for e in xml.iter('body')}
    source_xml=ET.fromstring(source_path.read_text(encoding='utf-8'))
    tendon_block=ET.SubElement(xml,'tendon');site_rows=[];tendon_names=[]
    for element in source_xml.findall('./tendon/spatial'):
        transferred=ET.SubElement(tendon_block,'spatial',name='candidate_'+element.get('name'),width='.005')
        tendon_names.append(transferred.get('name'))
        for site_ref in element:
            if site_ref.tag!='site':raise ValueError('Unsupported source wrap primitive')
            name=site_ref.get('site');sid=mj.mj_name2id(sm,mj.mjtObj.mjOBJ_SITE,name)
            parent=mj.mj_id2name(sm,mj.mjtObj.mjOBJ_BODY,int(sm.site_bodyid[sid]))
            target=destination[parent];key='combined' if parent in ('LFTrochanter','LFFemur') else parent
            so,sr,to,tr,scale=transforms[key]
            world=to+tr@((sr.T@(sd.site_xpos[sid]-so))*scale)
            target_id=body_id(tm,target);local=td.xmat[target_id].reshape(3,3).T@(world-td.xpos[target_id])
            site_name='candidate_'+name
            if not any(e.get('name')==site_name for e in bodies[target].findall('site')):
                ET.SubElement(bodies[target],'site',name=site_name,pos=' '.join(format(v,'.17g') for v in local),size='.007',rgba='1 .3 .1 1')
                site_rows.append({'source_site':name,'source_body':parent,'target_body':target,'source_frame':key,'scale':scale,'target_local':local.tolist()})
            ET.SubElement(transferred,'site',site=site_name)
    candidate_xml=ET.tostring(xml,encoding='unicode');model=mj.MjModel.from_xml_string(candidate_xml);data=mj.MjData(model)
    mj.mj_resetDataKeyframe(model,data,mj.mj_name2id(model,mj.mjtObj.mjOBJ_KEY,'neutral'));mj.mj_forward(model,data)
    tendons=[[mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,e.get('site')) for e in t] for t in tendon_block]
    neutral=data.qpos.copy();records=[];max_error=0.;max_base=0.
    for index,name in enumerate(tendon_names):
        tensions=np.zeros(len(tendons));tensions[index]=1.
        data.qpos[:]=neutral;mj.mj_forward(model,data);tau=pulling_force(model,data,tendons,tensions)
        max_base=max(max_base,float(np.max(np.abs(tau[:6]))))
        finite=np.zeros(model.nv)
        for j in body.active_joint_ids:
            q=model.jnt_qposadr[j];d=model.jnt_dofadr[j];eps=1e-6
            data.qpos[:]=neutral;data.qpos[q]+=eps;mj.mj_forward(model,data);plus=data.ten_length[index]
            data.qpos[:]=neutral;data.qpos[q]-=eps;mj.mj_forward(model,data);minus=data.ten_length[index]
            finite[d]=-(plus-minus)/(2*eps)
        error=float(np.max(np.abs(tau-finite)));max_error=max(max_error,error)
        records.append({'tendon':name,'unit_tension_torque':{body.joint_names[j]:float(tau[model.jnt_dofadr[j]]) for j in body.active_joint_ids if abs(tau[model.jnt_dofadr[j]])>1e-8},'finite_difference_error':error})
    if max_error>1e-7 or max_base>1e-10:raise ValueError('Force transfer fails virtual work/internal force check')
    out=ROOT/'runs/lf-tendon-candidate';out.mkdir(exist_ok=True)
    (out/'body.xml').write_text(candidate_xml,encoding='utf-8',newline='\n')
    result={'source_sha256':audit['source_sha256'],'base_body_sha256':body.digest,'candidate_xml_sha256':hashlib.sha256(candidate_xml.encode()).hexdigest(),
            'site_transfers':site_rows,'tendons':records,'max_virtual_work_error':max_error,'max_free_base_generalized_force':max_base,
            'status':'CALIBRATION_CANDIDATE','assumption':'Bone endpoint and bend-plane frame with uniform per-bone scaling; source thorax uses LF/RF/LM coxa landmarks.',
            'registration_is_biologically_validated':False,'CNS_executed':False,'walking_passed':False,'runtime_integrated':False}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('site_transfers','tendons')},indent=2))

if __name__=='__main__':main()
