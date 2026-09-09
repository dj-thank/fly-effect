"""Bounded same-leg connectivity correspondence, retained as an explicit prior."""
from pathlib import Path
import hashlib
import json
import numpy as np
import scipy.sparse as sp
from organism_core.body import Body
from organism_core.motor_map import MotorMap

def main():
    root=Path(__file__).resolve().parent
    graph=root.parent/'data/graph'
    ids=np.load(graph/'body_ids.npy');motor=np.load(graph/'motor_indices.npy')
    body=Body();mapping=MotorMap(ids,motor,body.active_joint_names)
    rows=[r for r in mapping.rows if r['required_for_walking']]
    indices=np.array([r['internal_index'] for r in rows])
    lookup=np.full(len(ids),-1,dtype=np.int32);lookup[indices]=np.arange(len(indices))
    dtype=np.dtype([('pre','<u4'),('post','<u4'),('count','<u4'),('source_row','<u8')])
    edges=np.memmap(graph/'edges.bin',mode='r',dtype=dtype)
    with (graph/'edges.bin').open('rb') as f:graph_hash=hashlib.file_digest(f,'sha256').hexdigest()
    if graph_hash!='45233dbc6a67b676bdf5a6b1ee0fc206bea5eae66074f92a09efcbb56aef89d5':raise ValueError('Graph changed')
    rr=[];cc=[];vv=[]
    for start in range(0,len(edges),1000000):
        chunk=edges[start:start+1000000];target=lookup[chunk['post']];valid=target>=0
        rr.append(target[valid]);cc.append(chunk['pre'][valid]);vv.append(chunk['count'][valid])
    matrix=sp.csr_matrix((np.concatenate(vv).astype(float),(np.concatenate(rr),np.concatenate(cc))),shape=(len(rows),len(ids)))
    norm=np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    normalized=sp.diags(1/np.maximum(norm,1e-30))@matrix
    similarity=(normalized@normalized.T).toarray()
    inferred=[]
    for i,row in enumerate(rows):
        if row['joint'] is not None:continue
        eligible=[j for j,other in enumerate(rows) if other['joint'] is not None and other['side']==row['side'] and other['neuromere']==row['neuromere']]
        ranked=sorted(eligible,key=lambda j:(-similarity[i,j],rows[j]['body_id']))[:3]
        candidates=[{'body_id':rows[j]['body_id'],'joint':rows[j]['joint'],'muscle_polarity':rows[j]['muscle_polarity'],
                     'cosine':float(similarity[i,j]),'type':rows[j]['type']} for j in ranked]
        accepted=bool(candidates and candidates[0]['cosine']>=.1)
        inferred.append({'body_id':row['body_id'],'original_status':row['status'],'original_type':row['type'],
                         'side':row['side'],'neuromere':row['neuromere'],'candidates':candidates,
                         'usable_as_candidate_prior':accepted,'biologically_confirmed':False})
    result={'method':'cosine of actual incoming synapse-count vectors; nearest named motor within same side and neuromere',
            'minimum_cosine':.1,'top_k':3,'status':'INFERRED mechanical correspondence; NOT measured muscle identity',
            'graph_sha256':graph_hash,'source_hashes':mapping.source_hashes,'neurons':inferred,
            'usable_count':sum(x['usable_as_candidate_prior'] for x in inferred),'unresolved_count':sum(not x['usable_as_candidate_prior'] for x in inferred),
            'graph_edges_modified':False}
    (root/'runs/motor-inference-v2.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'usable':result['usable_count'],'unresolved':result['unresolved_count'],
                      'minimum_best_cosine':min(x['candidates'][0]['cosine'] for x in inferred)},indent=2))

if __name__=='__main__':main()
