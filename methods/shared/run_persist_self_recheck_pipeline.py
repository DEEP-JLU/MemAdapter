from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; RESULTS=ROOT/'datasets'/'persistbench'/'results'; LOGS=RESULTS/'_run_logs'/'self_recheck_pipeline_20260921'
SYSTEMS=('AMEM','Mem0','naiveRAG','MemoryBank','LightMem'); SD={'AMEM':'a-mem','Mem0':'mem0','naiveRAG':'naive-rag','MemoryBank':'memory-bank','LightMem':'light-mem'}
SUBSETS=(('cross_domain',(1,2)),('sycophancy',(1,2)),('beneficial_memory_usage',()))
def env():
 e=os.environ.copy()
 for n in ('config/.external_bench_20260917.env','config/.post_retrieval.secret.env'):
  for l in (ROOT/n).read_text(encoding='utf-8-sig').splitlines():
   if '=' in l and not l.lstrip().startswith('#'):
    k,v=l.split('=',1);e[k.strip()]=v.strip().strip('"').strip("'")
 e.update(EXTERNAL_RESULTS_ROOT=str(RESULTS),NORMALIZED_RESULTS_LAYOUT='1',MEMSYCO_BENCHMARK=str(ROOT/'datasets'/'memsyco-bench'/'evaluation'),MEMADAPTER_ENABLE_REASONING='',MEMADAPTER_API_MODE='chat_completions',PYTHONIOENCODING='utf-8');return e
def d(system,subset,run):
 p=RESULTS/SD[system]/'self-recheck'/subset;return p if not run else p/f'run{run}'
def stage(system,kind):
 jobs=[]; e=env(); LOGS.mkdir(parents=True,exist_ok=True)
 if kind=='generate': items=[(u,r) for u,rs in SUBSETS for r in rs]
 else: items=[(u,r) for u,rs in (('cross_domain',(0,1,2)),('sycophancy',(0,1,2)),('beneficial_memory_usage',(0,))) for r in rs]
 workers=16
 for u,r in items:
  ret=ROOT/'datasets'/'persistbench'/'retrieval'/u/system/'retrieved_topk.jsonl'; out=d(system,u,r); stem=f'{system}_{kind}_{u}_run{r}'
  if kind=='generate': workers=4; cmd=[sys.executable,str(ROOT/'methods'/'shared'/'run_extra_interventions.py'),'--system',system,'--retrieval-file',str(ret),'--dataset','persistbench','--method','self_recheck','--output-dir',str(out),'--run-index',str(r),'--workers',str(workers),'--continue-on-error','--retry-failures']
  else: cmd=[sys.executable,'-m','external_benchmarks.run_external_judge','--retrieval-file',str(ret),'--system',system,'--arm','self_recheck','--run-index',str(r),'--workers',str(workers),'--continue-on-error']
  so=(LOGS/f'{stem}.stdout.log').open('w',encoding='utf8');se=(LOGS/f'{stem}.stderr.log').open('w',encoding='utf8'); proc=subprocess.Popen(cmd,cwd=ROOT,env=e,stdout=so,stderr=se); jobs.append(proc); code=proc.wait();
  if code: return 1
 return 0
def main():
 system=sys.argv[1] if len(sys.argv)>1 else None
 if system:
  rc=stage(system,'generate')
  if not rc: rc=stage(system,'judge')
  return rc
 LOGS.mkdir(parents=True,exist_ok=True);e=env();ps=[subprocess.Popen([sys.executable,__file__,s],cwd=ROOT,env=e,stdout=(LOGS/f'{s}_controller.stdout.log').open('w'),stderr=(LOGS/f'{s}_controller.stderr.log').open('w')) for s in SYSTEMS];(LOGS/'pids.json').write_text(json.dumps({'pids':[p.pid for p in ps]}));print(json.dumps({'started':len(ps),'pids':[p.pid for p in ps],'logs':str(LOGS)}));return 0
if __name__=='__main__':raise SystemExit(main())
