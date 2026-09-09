"""Array/tree checkpoint codec without pickle or executable deserialization."""
from pathlib import Path
import hashlib
import json
import os
import uuid
import numpy as np


def save(path, state, identity):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    arrays={}
    def encode(value):
        if isinstance(value,np.ndarray):
            if value.dtype.hasobject:raise TypeError('Object arrays forbidden')
            name=f'a{len(arrays)}';arrays[name]=value
            return {'array':name}
        if isinstance(value,np.generic):return encode(value.item())
        if isinstance(value,dict):
            if not all(isinstance(k,str) for k in value):raise TypeError('String keys required')
            return {'dict':[[k,encode(v)] for k,v in value.items()]}
        if isinstance(value,(list,tuple)):
            return {'tuple' if isinstance(value,tuple) else 'list':[encode(v) for v in value]}
        if value is None or isinstance(value,(bool,int,float,str)):
            return {'scalar':value}
        raise TypeError('Unsupported checkpoint type: '+str(type(value)))
    tree=encode(state)
    arrays['metadata']=np.array(json.dumps({'schema':1,'identity':identity,'tree':tree},allow_nan=False))
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('xb') as f:np.savez_compressed(f,**arrays)
    os.replace(temp,path)
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(path),'sha256':digest}


def load(path,identity,expected_sha256=None):
    path=Path(path)
    if expected_sha256:
        with path.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=expected_sha256:raise ValueError('Checkpoint hash mismatch')
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        if meta.get('schema')!=1 or meta.get('identity')!=identity:
            differences=[k for k in set(meta.get('identity',{}))|set(identity) if meta.get('identity',{}).get(k)!=identity.get(k)]
            raise ValueError('Checkpoint identity mismatch: '+','.join(sorted(differences)))
        def decode(node):
            if len(node)!=1:raise ValueError('Invalid checkpoint node')
            tag,value=next(iter(node.items()))
            if tag=='array':return data[value].copy()
            if tag=='scalar':return value
            if tag=='tuple':return tuple(decode(v) for v in value)
            if tag=='list':return [decode(v) for v in value]
            if tag=='dict':return {k:decode(v) for k,v in value}
            raise ValueError('Unsupported checkpoint node')
        return decode(meta['tree'])
