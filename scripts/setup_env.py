"""Create the GeoFuse conda environment and install GeoFuse into it.

    python scripts/setup_env.py [--yes] [--torch auto|cuda|cpu] [--env-name NAME]
                                [--log FILE]

``install.bat`` / ``install.sh`` pass their arguments through. An existing
environment of the same name is replaced only after confirmation (``--yes``
confirms in advance, for unattended runs).
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile

# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────

ENV_NAME = "geofuse"
PYTHON_VERSION = "3.13"

# Conda-forge packages, solved together with Python in one step so that the
# BLAS choice, GDAL and every compiled library stay consistent. Everything
# conda-forge carries for Windows, Linux and macOS comes from here; pip only
# adds what it lacks (below), and leaves these in place.
CONDA_PACKAGES = [
    "libblas=*=*openblas",
    "libcblas=*=*openblas",
    "liblapack=*=*openblas",
    "numpy=2.4.6",
    "scipy=1.17.1",
    # PyTorch, installed by pip afterwards, needs setuptools < 82
    "setuptools=81",
    # Geospatial stack
    "gdal=3.12.3",
    "geopandas=1.1.4",
    "rasterio=1.5.0",
    "shapely=2.1.2",
    "fiona=1.10.1",
    # streamlit needs pandas < 3
    "pandas=2.3.3",
    "statsmodels=0.14.6",
    "matplotlib=3.10.8",
    "scikit-learn=1.6.0",
    "pillow=12.0.0",
    "tqdm=4.67.1",
    # Async HTTP — used by the in-house geofuse.streetview client (GVIEngine)
    "aiohttp=3.12.15",
    # Bayesian index diagnostics
    "arviz=1.2.0",
    # Distance-correlation objectives and spline residualization
    "dcor=0.7",
    "patsy=1.0.2",
    # UI / Web
    "streamlit=1.52.1",
    "streamlit-folium=0.27.4",
    "folium=0.20.0",
    "plotly=6.7.0",
    # Earth Engine
    "earthengine-api=1.7.4",
    # Command line: monitor, credential store, per-user folders, resource
    # locks, memory sizing
    "textual=8.2.8",
    "keyring=25.7.0",
    "platformdirs=4.9.6",
    "filelock=3.29.0",
    "psutil=7.2.2",
    # Tests
    "pytest=9.1.1",
    "optuna=4.6.0",
    # Dev tooling — must match the versions in .github/workflows/code-quality.yml
    "black=25.1.0",
    "ruff=0.15.12",
    "isort=8.0.1",
]

# PyTorch wheels: conda-forge has no CUDA build for Windows. "auto" takes the
# CUDA build on Windows and Linux (it runs on the CPU too, but is several GB
# larger) and the standard build on macOS.
TORCH_PACKAGES = "torch==2.11.0 torchvision==0.26.0"
TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu128",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

# What conda-forge cannot provide alongside the packages above.
PIP_PACKAGES = [
    # Bayesian index sampler. conda-forge has no jaxlib for Windows, and its
    # Linux / macOS jaxlib for Python 3.13 needs protobuf 7, which
    # streamlit 1.52 refuses; the pip wheels carry their own libraries.
    "jax==0.11.0",
    "numpyro==0.21.0",
    # Not on conda-forge
    "streamlit-sortables==0.3.1",
]

# Imported after the install, in this order; a failing one fails the
# installation. Pillow comes before torchvision, as in geofuse.vision: on
# Windows, torchvision's own jpeg8.dll breaks a later conda-forge Pillow.
VERIFY_MODULES = [
    "geofuse",
    "geofuse.cli",
    "geofuse.bayesian_index",
    "PIL.Image",
    "torch",
    "torchvision",
    "geofuse.vision",
    "jax",
    "numpyro",
    "arviz",
    "dcor",
    "geopandas",
    "rasterio",
    "ee",
    "streamlit",
    "plotly",
    "psutil",
]

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ────────────────────────────────────────────────────────────────────
# Commands
# ────────────────────────────────────────────────────────────────────


def run_cmd(command, log_file=None, optional=False):
    """Run a shell command, streaming its output to the console and the log.

    On a non-zero exit the last 40 lines are printed again, so the reason is
    visible even in a long CI log, and the installer stops. With
    ``optional=True`` it prints a warning and carries on instead.
    """
    output_lines = []
    log = open(log_file, "a", encoding="utf-8") if log_file else None
    try:
        if log:
            log.write(f"\n$ {command}\n")
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in process.stdout:
            output_lines.append(line)
            print(line, end="", flush=True)
            if log:
                log.write(line)
                log.flush()
        process.wait()
    finally:
        if log:
            log.close()

    if process.returncode == 0:
        return
    tail = output_lines[-40:]
    if optional:
        print(f"\n[WARN] Optional command failed (continuing): {command}")
    else:
        print(
            f"\n[ERROR] Command failed with exit code {process.returncode}: {command}"
        )
    if tail:
        print("\n--- last output ---")
        for ln in tail:
            print(ln, end="")
        print("--- end ---\n")
    if not optional:
        sys.exit(1)


def conda_json(*args):
    """Run ``conda <args> --json`` and return the parsed output."""
    out = subprocess.run(
        "conda " + " ".join(args) + " --json",
        shell=True,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


def get_solver():
    return "mamba" if shutil.which("mamba") else "conda"


def env_prefix(env_name):
    """The prefix of the environment called exactly ``env_name``, or None."""
    info = conda_json("info")
    root = os.path.normcase(os.path.normpath(info["root_prefix"]))
    for prefix in conda_json("env", "list")["envs"]:
        norm = os.path.normcase(os.path.normpath(prefix))
        if norm != root and os.path.basename(norm) == os.path.normcase(env_name):
            return prefix
    return None


def env_python(prefix):
    if platform.system() == "Windows":
        return os.path.join(prefix, "python.exe")
    return os.path.join(prefix, "bin", "python")


def confirm_replace(env_name, prefix, assume_yes):
    """Whether an existing environment may be removed and built again."""
    if assume_yes:
        return True
    if not sys.stdin or not sys.stdin.isatty():
        print(
            f"[ERROR] Environment '{env_name}' already exists at {prefix}. "
            "Run the installer with --yes to replace it."
        )
        return False
    print(
        f"[WARN] Environment '{env_name}' already exists at {prefix}.\n"
        "       Replacing it removes everything installed in it, including "
        "packages added by hand."
    )
    answer = input("       Remove it and install afresh? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


def torch_command(python, choice):
    """The pip command that installs PyTorch for this platform and choice."""
    system = platform.system()
    base = f'"{python}" -m pip install {TORCH_PACKAGES} --no-warn-script-location'
    if system == "Darwin":
        if choice == "cuda":
            print("[ERROR] There is no CUDA build of PyTorch for macOS.")
            sys.exit(1)
        return base
    index = TORCH_INDEX["cpu" if choice == "cpu" else "cuda"]
    return f"{base} --index-url {index}"


def pip_replaced_conda(python):
    """Conda-pinned packages that a pip step has since replaced.

    Read from each package's ``INSTALLER`` record: ``conda list`` reports
    packages of environments created by mamba 2 as pip installs.
    """
    names = [
        re.split(r"[=<>!]", p)[0] for p in CONDA_PACKAGES if not p.startswith("lib")
    ]
    script = "\n".join(
        [
            "import importlib.metadata as md, json",
            "out = []",
            f"for name in {names!r}:",
            "    try:",
            "        who = md.distribution(name).read_text('INSTALLER') or ''",
            "    except md.PackageNotFoundError:",
            "        continue",
            "    if who.strip() not in ('', 'conda'):",
            "        out.append(name)",
            "print(json.dumps(out))",
        ]
    )
    res = subprocess.run([python, "-I", "-c", script], capture_output=True, text=True)
    if res.returncode != 0 or not res.stdout.strip():
        return []
    return json.loads(res.stdout.strip().splitlines()[-1])


def verify(python, log_file):
    """Import every key module and report the versions and the accelerator."""
    script = "\n".join(
        [
            "import importlib, platform",
            f"for name in {VERIFY_MODULES!r}:",
            "    importlib.import_module(name)",
            "import geofuse, jax, numpyro, torch",
            "print(f'   [OK] GeoFuse package: {geofuse.__file__}')",
            "if platform.system() == 'Darwin':",
            "    acc = f'MPS available: {torch.backends.mps.is_available()}'",
            "else:",
            "    acc = f'CUDA available: {torch.cuda.is_available()}'",
            "print(f'   [OK] PyTorch {torch.__version__} - {acc}')",
            "print(f'   [OK] JAX {jax.__version__}, NumPyro {numpyro.__version__}')",
            f"print('   [OK] Imported {len(VERIFY_MODULES)} modules')",
        ]
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
    try:
        # -I keeps the script's folder off sys.path, where a stray "geofuse"
        # folder would shadow the installed package.
        run_cmd(f'"{python}" -I "{f.name}"', log_file)
    finally:
        os.remove(f.name)


# ────────────────────────────────────────────────────────────────────
# Installation
# ────────────────────────────────────────────────────────────────────


def main(env_name, log_file=None, assume_yes=False, torch_choice="auto"):
    system = platform.system()
    solver = get_solver()

    try:
        conda_base = conda_json("info")["root_prefix"]
        os.environ["MAMBA_ROOT_PREFIX"] = conda_base
        print(f"[INFO] MAMBA_ROOT_PREFIX -> {conda_base}")
    except Exception as exc:
        print(f"[WARN] Could not detect conda base, mamba may fail: {exc}")

    if not log_file:
        log_dir = os.path.join(ROOT_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "install.log")
    if os.path.exists(log_file):
        os.remove(log_file)

    print(f"\n[INFO] Starting setup on {system} using {solver}...")
    print(f"[INFO] Detailed logs: {log_file}\n")

    # ── 1. Environment with Python and the conda packages ────────────
    existing = env_prefix(env_name)
    if existing:
        if not confirm_replace(env_name, existing, assume_yes):
            print("[INFO] Nothing was changed.")
            sys.exit(1)
        print(f"[INFO] Removing environment '{env_name}'...")
        run_cmd(f"conda env remove -n {env_name} -y", log_file)

    print(f"[1/5] Creating '{env_name}' with Python {PYTHON_VERSION} ({solver})...")
    conda_str = " ".join(f'"{p}"' for p in CONDA_PACKAGES)
    run_cmd(
        f"{solver} create -n {env_name} -y --override-channels -c conda-forge "
        f"python={PYTHON_VERSION} pip {conda_str}",
        log_file,
    )
    prefix = env_prefix(env_name)
    if not prefix:
        print(f"[ERROR] Environment '{env_name}' was not created.")
        sys.exit(1)
    python = env_python(prefix)
    print(f"[INFO] Environment: {prefix}\n")

    # ── 2. PyTorch ───────────────────────────────────────────────────
    print(f"[2/5] Installing PyTorch ({torch_choice})...")
    run_cmd(torch_command(python, torch_choice), log_file)

    # ── 3. Python libraries ──────────────────────────────────────────
    print("[3/5] Installing Python libraries...")
    run_cmd(
        f'"{python}" -m pip install --upgrade pip --no-warn-script-location',
        log_file,
    )
    pip_str = " ".join(PIP_PACKAGES)
    run_cmd(f'"{python}" -m pip install {pip_str} --no-warn-script-location', log_file)

    # ── 4. GeoFuse itself ────────────────────────────────────────────
    print("[4/5] Installing GeoFuse (editable)...")
    run_cmd(
        f'"{python}" -m pip install -e "{ROOT_DIR}" --no-warn-script-location',
        log_file,
    )

    # ── 5. Verification ──────────────────────────────────────────────
    print("[5/5] Verifying the installation...")
    replaced = pip_replaced_conda(python)
    if replaced:
        print(
            "[WARN] pip replaced these conda packages: "
            + ", ".join(replaced)
            + ". See docs/TROUBLESHOOTING.md if imports fail with a DLL error."
        )
    verify(python, log_file)
    # The readiness check needs the model file and a sign-in, which may come
    # after installing, so its problems do not fail the install.
    run_cmd(f'"{python}" -m geofuse doctor --offline', log_file, optional=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create the GeoFuse conda environment and install GeoFuse into it."
    )
    parser.add_argument(
        "log_file", nargs="?", default=None, help="log file (default logs/install.log)"
    )
    parser.add_argument("--log", dest="log_option", help="log file")
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="replace an existing environment without asking",
    )
    parser.add_argument(
        "--torch",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="PyTorch build: auto (CUDA on Windows/Linux), cuda or cpu",
    )
    parser.add_argument("--env-name", default=ENV_NAME, help="environment name")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.env_name,
        log_file=args.log_option or args.log_file,
        assume_yes=args.yes,
        torch_choice=args.torch,
    )
