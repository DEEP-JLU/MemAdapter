from __future__ import annotations
import os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; RESULTS=ROOT/'datasets'/'persistbench'/'results'; LOGS=RESULTS/'_run_logs'/'remote_dynamic_memgate_20260921'; SYSTEMS=('AMEM','Mem0','naiveRAG','MemoryBank','LightMem'); SD={'AMEM':'a-mem','Mem0':'mem0','naiveRAG':'naive-rag','MemoryBank':'memory-bank','LightMem':'light-mem'}; METHODS=('dynamic_partition','memgate'); CELLS=(('cross_domain',0),('cross_domain',1),('cross_domain',2),('sycophancy',0),('sycophancy',1),('sycophancy',2),('beneficial_memory_usage',0))
def run(s,m):
 env=os.environ.copy(); env.update(EXTERNAL_RESULTS_ROOT=str(RESULTS),NORMALIZED_RESULTS_LAYOUT='1',MEMSYCO_BENCHMARK=str(ROOT/'datasets'/'memsyco-bench'/'evaluation'),MEMADAPTER_ENABLE_REASONING='',MEMADAPTER_API_MODE='chat_completions')
 for u,r in CELLS:
  ret=ROOT/'datasets'/'persistbench'/'retrieval'/u/s/'retrieved_topk.jsonl'; stem=f'{s}_{m}_{u}_run{r}.retryjudge.log'; cmd=[sys.executable,'-m','external_benchmarks.run_external_judge','--retrieval-file',str(ret),'--system',s,'--arm',m,'--run-index',str(r),'--workers','12','--continue-on-error']
  with (LOGS/stem).open('w') as f:
   if subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode:return 1
 return 0
if __name__=='__main__':
 if len(sys.argv)==3: raise SystemExit(run(sys.argv[1],sys.argv[2]))
 LOGS.mkdir(parents=True,exist_ok=True)
 for s in SYSTEMS:
  for m in METHODS: subprocess.Popen([sys.executable,__file__,s,m],cwd=ROOT)
