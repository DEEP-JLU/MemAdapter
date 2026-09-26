"""Pinned environment entry point for the existing external MemAdapter/judge harness.

Set EXTERNAL_RESULTS_ROOT to a separate smoke root before invoking for smoke runs.
Generate: pass native run_memadapter arguments after 'generate'.
Judge: pass run_external_judge arguments after 'judge'.
"""
import sys
import run_extra_interventions as shared

if __name__ == '__main__':
    shared.load_environment()
    mode = sys.argv.pop(1)
    if mode == 'generate':
        sys.argv.insert(1, 'generate')
        shared.native.main()
    elif mode == 'judge':
        from external_benchmarks.run_external_judge import main
        raise SystemExit(main())
    else:
        raise SystemExit('Expected generate or judge')
