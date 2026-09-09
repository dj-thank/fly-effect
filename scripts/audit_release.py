"""Check the publication boundary; this is not a complete secret/security scanner."""
from pathlib import Path
import re,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','.venv','venv','work','runs','cache','data','dist','build','__pycache__','.pytest_cache'}
FORBIDDEN={'.npz','.npy','.parquet','.bin','.pem','.key'}
def main():
    result=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True)
    listed=[ROOT/x for x in result.stdout.decode('utf-8').split('\0') if x] if result.returncode==0 else []
    files=listed or [p for p in ROOT.rglob('*') if p.is_file() and not set(p.relative_to(ROOT).parts)&EXCLUDED and not any(x.endswith('.egg-info') for x in p.parts)]
    failures=[]
    patterns=[re.compile(r'[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+'),re.compile(r'ghp_[A-Za-z0-9]{30,}'),re.compile(r'github_pat_[A-Za-z0-9_]{30,}'),re.compile(r'sk-proj-[A-Za-z0-9_-]{30,}')]
    for p in files:
        rel=p.relative_to(ROOT)
        if p.suffix in FORBIDDEN or rel.parts[0] in EXCLUDED or p.name.startswith('.env'):failures.append(str(rel)+': forbidden release file');continue
        if p.stat().st_size>2_000_000:failures.append(str(rel)+': oversized file');continue
        try:text=p.read_text(encoding='utf-8-sig')
        except UnicodeDecodeError:failures.append(str(rel)+': unreviewed binary');continue
        if any(x.search(text) for x in patterns):failures.append(str(rel)+': possible secret or personal path')
        if p.suffix=='.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
                target=target.split('#')[0]
                if not target or '://' in target or target.startswith('mailto:'):continue
                if not (p.parent/target).exists():failures.append(str(rel)+': broken link '+target)
    for required in ['README.md','README.en.md','LICENSE','CONTRIBUTING.md','GOVERNANCE.md','SECURITY.md','THIRD_PARTY_NOTICES.md','docs/STATUS.md']:
        if not (ROOT/required).is_file():failures.append('missing '+required)
    print('Publication boundary:',len(files),'files checked')
    for problem in failures:print(problem)
    if failures:raise SystemExit(1)
    print('PASS — still requires human review of rights, claims, and release contents.')

if __name__=='__main__':main()

