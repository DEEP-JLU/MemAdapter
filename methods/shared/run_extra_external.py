"""Public entry point for the post-retrieval intervention runner.

The local environment supplies credentials and endpoint URLs.  ``generate``
accepts the arguments documented by :mod:`run_extra_interventions`; ``judge``
forwards its arguments to the external benchmark judge.
"""
import sys

try:  # Works both as ``python -m`` and as a direct script.
    from . import run_extra_interventions as shared
except ImportError:
    import run_extra_interventions as shared


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("Expected 'generate' or 'judge'")
    mode = argv.pop(0)
    if mode == 'generate':
        shared.main(argv)
    elif mode == 'judge':
        shared.configure_public_environment()
        from external_benchmarks.run_external_judge import main
        raise SystemExit(main(argv))
    else:
        raise SystemExit('Expected generate or judge')


if __name__ == '__main__':
    main()
