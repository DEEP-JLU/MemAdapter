# Dynamic Partition

Implemented in `../shared/run_extra_interventions.py`. Its partition prompt and
ID-preservation validation operate only on the current request and frozen
retrieved memories; dialogue context, query-session history, and task evidence
are not inputs.
