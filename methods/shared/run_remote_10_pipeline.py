from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; RESULTS=ROOT/'datasets'/'persistbench'/'results'; LOGS=RESULTS/'_run_logs'/'remote_dynamic_memgate_20260921'
SYSTEMS=('AMEM','Mem0','naiveRAG','MemoryBank','LightMem'); SD={'AMEM':'a-mem','Mem0':'mem0','naiveRAG':'naive-rag','MemoryBank':'memory-bank','LightMem':'light-mem'}; METHODS=('dynamic_partition','memgate'); WORKERS=12
CELLS=(('cross_domain',0),('cross_domain',1),('cross_domain',2),('sycophancy',0),('sycophancy',1),('sycophancy',2),('beneficial_memory_usage',0))
def run(system,method):
 env=os.environ.copy(); env.update(EXTERNAL_RESULTS_ROOT=str(RESULTS),NORMALIZED_RESULTS_LAYOUT='1',MEMSYCO_BENCHMARK=str(ROOT/'datasets'/'memsyco-bench'/'evaluation'),MEMADAPTER_ENABLE_REASONING='',MEMADAPTER_API_MODE='chat_completions',PYTHONIOENCODING='utf-8')
 for subset,run in CELLS:
  ret=ROOT/'datasets'/'persistbench'/'retrieval'/subset/system/'retrieved_topk.jsonl'; out=RESULTS/SD[system]/method.replace('_','-')/subset/(f'run{run}' if run else '')
  out.mkdir(parents=True,exist_ok=True)
  stem=f'{system}_{method}_{subset}_run{run}'
  gen=[sys.executable,str(ROOT/'methods'/'shared'/'run_extra_interventions.py'),'--system',system,'--retrieval-file',str(ret),'--dataset','persistbench','--method',method,'--output-dir',str(out),'--run-index',str(run),'--workers',str(WORKERS),'--continue-on-error','--retry-failures']
  with (LOGS/f'{stem}.generate.log').open('w') as f:
   if subprocess.run(gen,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode: return 1
  judge=[sys.executable,'-m','external_benchmarks.run_external_judge','--retrieval-file',str(ret),'--system',system,'--arm',method,'--run-index',str(run),'--workers',str(WORKERS),'--continue-on-error']
  with (LOGS/f'{stem}.judge.log').open('w') as f:
   if subprocess.run(judge,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode: return 1
 return 0
def main():
 LOGS.mkdir(parents=True,exist_ok=True); procs=[]
 for s in SYSTEMS:
  for m in METHODS:
   so=(LOGS/f'{s}_{m}.controller.log').open('w'); procs.append(subprocess.Popen([sys.executable,__file__,s,m],stdout=so,stderr=subprocess.STDOUT))
 if len(sys.argv)==3: return run(sys.argv[1],sys.argv[2])
 (LOGS/'pids.json').write_text(json.dumps({'pids':[p.pid for p in procs],'workers_per_task':WORKERS,'tasks':10})); print(json.dumps({'started':10,'workers_per_task':WORKERS,'logs':str(LOGS)})); return 0
if __name__=='__main__':
 if len(sys.argv)==3: raise SystemExit(run(sys.argv[1],sys.argv[2]))
 raise SystemExit(main())
