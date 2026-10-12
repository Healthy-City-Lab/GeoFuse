"""``geofuse hpc script``: batch scripts for a cluster scheduler.

::

    geofuse hpc script RUN.toml [--setup "module load ...; conda activate geofuse"]
        [--account A] [--partition P] [--gpu-partition P] [--gpu-gres gpu:1]
        [--time 24:00:00] [--cpus 8] [--mem 32G] [--fusion-cpus 32]
        [--fusion-mem 128G] [-o DIR]

A run becomes one SLURM job array per dependency level and resource group —
``gpu`` for GVI, ``fusion`` for fusion fits, ``light`` for NDVI downloads and
GVI merges. Each array task runs one job (``geofuse run RUN --level K --group
G --array-task $SLURM_ARRAY_TASK_ID``), and ``submit.sh`` submits the arrays
so every level waits for the one before (``afterok``). All tasks write to one
state folder, so ``geofuse status`` / ``geofuse watch`` on the login node
follow them. Nothing is specific to one cluster: the flags are plain SLURM,
and ``--setup`` holds whatever has to happen before ``geofuse`` runs.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from geofuse.cli.commands import EXIT_OK, CliError, _add_run_args, _load, _state

_GROUP_TEXT = {
    "gpu": "GVI (needs a GPU)",
    "fusion": "fusion fits (cores and memory)",
    "light": "NDVI downloads and GVI merges",
}


def _forwarded(args) -> list[str]:
    """The command-line choices every task must repeat to plan the same jobs."""
    argv: list[str] = []
    if getattr(args, "config", None):
        argv += ["--config", str(Path(args.config).resolve())]
    for step in args.only:
        argv += ["--only", step]
    for raw in args.overrides:
        argv += ["--set", raw]
    for name, value in vars(args).items():
        if not name.startswith("setting__") or value is None:
            continue
        flag = "--" + name[len("setting__") :].replace("_", "-")
        if isinstance(value, bool):
            argv.append(flag if value else "--no-" + flag[2:])
        elif isinstance(value, list):
            argv += [flag, *map(str, value)]
        else:
            argv += [flag, str(value)]
    return argv


def _resources(args, group: str) -> list[str]:
    if group == "gpu":
        lines = [f"--cpus-per-task={args.cpus}", f"--mem={args.mem}"]
        if args.gpu_gres:
            lines.append(f"--gres={args.gpu_gres}")
        partition = args.gpu_partition or args.partition
    elif group == "fusion":
        lines = [f"--cpus-per-task={args.fusion_cpus}", f"--mem={args.fusion_mem}"]
        partition = args.partition
    else:
        lines = ["--cpus-per-task=2", "--mem=8G"]
        partition = args.partition
    if partition:
        lines.append(f"--partition={partition}")
    if args.account:
        lines.append(f"--account={args.account}")
    return lines


def write_scripts(args, run, specs) -> tuple[Path, list[Path]]:
    """Write the array scripts and ``submit.sh``; returns the folder and the
    scripts in submission order."""
    from geofuse.jobs.plans import job_group, levels

    stages: dict[tuple[int, str], int] = {}
    for spec, level in zip(specs, levels(specs), strict=True):
        key = (level, job_group(spec))
        stages[key] = stages.get(key, 0) + 1
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", run.name).strip("-") or "run"
    folder = Path(args.output or f"hpc-{slug}").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    state = _state(args).root
    command = shlex.split(args.geofuse)
    forwarded = _forwarded(args)
    setup = [line.strip() for line in (args.setup or "").split(";") if line.strip()]

    scripts: list[Path] = []
    for (level, group), count in sorted(stages.items()):
        name = f"level{level}-{group}"
        task = [
            *command,
            "--state-dir",
            str(state),
            "run",
            str(Path(run.path).resolve()),
            "--level",
            str(level),
            "--group",
            group,
            *forwarded,
        ]
        header = [
            "#!/bin/bash",
            f"# {run.name}: {count} job(s) at level {level}, {_GROUP_TEXT[group]}.",
            f"#SBATCH --job-name=geofuse-{slug}-{name}",
            f"#SBATCH --array=0-{count - 1}",
            f"#SBATCH --time={args.time}",
            "#SBATCH --output=logs/%x-%A_%a.out",
            *(f"#SBATCH {line}" for line in _resources(args, group)),
        ]
        body = [
            "set -euo pipefail",
            *setup,
            " ".join(shlex.quote(part) for part in task)
            + ' --array-task "${SLURM_ARRAY_TASK_ID}"',
        ]
        path = folder / f"{name}.sbatch"
        text = "\n".join(header + [""] + body) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")
        scripts.append(path)

    by_level: dict[int, list[Path]] = {}
    for (level, _group), path in zip(sorted(stages), scripts, strict=True):
        by_level.setdefault(level, []).append(path)
    lines = [
        "#!/bin/bash",
        f"# Submits run {run.name!r}: each level's arrays wait for every array of",
        "# the level before (afterok).",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        "mkdir -p logs",
        'after=""',
    ]
    for level in sorted(by_level):
        lines += ["", f"# level {level}", "ids=()"]
        for path in by_level[level]:
            lines += [
                f'id=$(sbatch --parsable ${{after:+--dependency=afterok:$after}} '
                f"{path.name})",
                'ids+=("${id%%;*}")',
                f'echo "{path.name}: $id"',
            ]
        lines.append('after=$(IFS=:; echo "${ids[*]}")')
    submit = folder / "submit.sh"
    submit.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return folder, scripts


def cmd_hpc(args) -> int:
    run, specs = _load(args)
    if run.path is None:
        raise CliError("hpc script needs a run file on disk.")
    folder, scripts = write_scripts(args, run, specs)
    print(f"Wrote {len(scripts)} array script(s) and submit.sh to {folder}:")
    for path in scripts:
        print(f"  {path.name}")
    print(
        f"\nSubmit with:  bash {folder / 'submit.sh'}\n"
        f"Follow with:  geofuse --state-dir {_state(args).root} watch"
    )
    return EXIT_OK


def add_parsers(sub, common) -> None:
    """This module's commands on ``sub`` (which adds ``common`` itself)."""
    p = sub.add_parser("hpc", help="batch scripts for a cluster scheduler")
    actions = p.add_subparsers(dest="action", metavar="ACTION", required=True)
    s = actions.add_parser(
        "script", parents=[common], help="SLURM job arrays for a run file"
    )
    _add_run_args(s, arrays=False)
    s.add_argument("--scheduler", choices=["slurm"], default="slurm")
    s.add_argument("-o", "--output", help="folder for the scripts (default hpc-RUN)")
    s.add_argument(
        "--setup",
        help="shell lines to run first, ';'-separated (load modules, activate "
        "the environment)",
    )
    s.add_argument(
        "--geofuse", default="geofuse", help="how to start geofuse on the nodes"
    )
    s.add_argument("--account")
    s.add_argument("--partition", help="for every array but GVI's (see --gpu-...)")
    s.add_argument("--gpu-partition", help="for the GVI arrays")
    s.add_argument("--gpu-gres", default="gpu:1", help="'' for no GPU request")
    s.add_argument("--time", default="24:00:00", help="per task, HH:MM:SS")
    s.add_argument("--cpus", type=int, default=8, help="per GVI task")
    s.add_argument("--mem", default="32G", help="per GVI task")
    s.add_argument("--fusion-cpus", type=int, default=32)
    s.add_argument("--fusion-mem", default="128G")
    p.set_defaults(func=cmd_hpc)
