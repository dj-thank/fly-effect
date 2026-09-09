"""Validate the minimal structure of an experiment record, not its scientific truth."""
import argparse,json
from pathlib import Path
def validate(record):
    for key in ('schema_version','kind','status','question','hypothesis','seed','units','budget','controls','success_criterion','falsification','evidence','biological_validation'):
        if key not in record:raise ValueError('Missing '+key)
    if record['schema_version']!=1:raise ValueError('Unknown schema')
    if record['kind'] not in ('synthetic','reference','organism_candidate'):raise ValueError('Unknown experiment kind')
    if record['status'] not in ('planned','completed','failed'):raise ValueError('Unknown status')
    if record['budget']['wall_seconds']<=0 or record['budget']['memory_mib']<=0:raise ValueError('Positive resource budget required')
    if record['status']=='completed' and not record['evidence']:raise ValueError('Completed experiment needs evidence locators')
    if record['kind']=='synthetic' and record['biological_validation']:raise ValueError('Synthetic experiment cannot claim biological validation')
def main():
    p=argparse.ArgumentParser();p.add_argument('record',type=Path);args=p.parse_args()
    validate(json.loads(args.record.read_text(encoding='utf-8')))
    print('Structure valid; evidence and scientific claims require review.')
if __name__=='__main__':main()

