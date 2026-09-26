"""Run a command with the frozen protocol environment in place.

Both arms of the experiment have to run under one environment, and only one of them
was getting it. ``build_retrieval`` and ``run_external_baseline`` call
``load_env_file()`` themselves; ``MemAdapter/run_memadapter.py`` does not, and must
not be modified. Invoked directly it therefore inherits whatever the shell happens
to hold, and the failure is not a clean "missing key" -- ``ModelClient("DeepSeek-V4-Flash")``
falls back from ``DEEPSEEK_API_KEY`` to ``OPENAI_API_KEY``, so an unrelated gateway
credential in the environment produces a 401 INVALID_API_KEY that reads like a
broken endpoint.

Rather than repeat the loader at every call site, this wrapper pins the environment
once and hands it to a child process:

    python -m external_benchmarks.with_protocol_env -- <command> [args...]

The shell still wins over the env file, because that is what ``load_env_file``
documents and what an operator overriding a value expects. What this adds is that
the override stops being invisible: keys the file pins differently from the live
environment are named on stderr and the run is refused unless ``--accept-live-env``
says the difference is deliberate. Only variable *names* are ever printed.
"""

from __future__ import annotations

import os
import subprocess
import sys

from . import run_config


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    accept_live = "--accept-live-env" in argv
    if accept_live:
        argv.remove("--accept-live-env")
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2

    # Check before loading: load_env_file only sets what is absent, so a conflicting
    # name survives it and would otherwise go unnoticed for the whole run.
    conflicts = run_config.env_file_conflicts()
    if conflicts and not accept_live:
        print(
            "[protocol-env] the live environment disagrees with "
            f"{run_config.DEFAULT_ENV_FILE} on: {', '.join(conflicts)}\n"
            "[protocol-env] the live value would win, silently changing the protocol. "
            "Re-run with --accept-live-env if that override is intended, or unset the "
            "variable(s). Values are not printed.",
            file=sys.stderr,
        )
        return 2
    if conflicts:
        print(
            f"[protocol-env] proceeding with a deliberate override of: {', '.join(conflicts)}",
            file=sys.stderr,
        )

    applied = run_config.load_env_file()
    run_config.export_protocol_environment()

    model = os.environ.get("DEEPSEEK_MODEL", "")
    base = os.environ.get("DEEPSEEK_BASE_URL", "")
    # Names only: DEEPSEEK_API_KEY's value is a credential and is never printed.
    key_source = (
        "DEEPSEEK_API_KEY" if os.environ.get("DEEPSEEK_API_KEY")
        else ("OPENAI_API_KEY" if os.environ.get("OPENAI_API_KEY") else "none")
    )
    print(
        f"[protocol-env] loaded {len(applied)} pin(s) from {run_config.DEFAULT_ENV_FILE}; "
        f"model={model!r} base_url={base!r} api_key_from={key_source} "
        f"temperature={run_config.resolved_temperature()} "
        f"max_tokens={run_config.resolved_max_tokens()} "
        f"thinking={run_config.resolved_thinking()}",
        file=sys.stderr,
    )

    return subprocess.call(argv, env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
