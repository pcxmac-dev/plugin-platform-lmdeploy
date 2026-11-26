"""What Install brings, what is installed here and what this system lacks (plugin-platform-lmdeploy 0.1.0).

The pins come from what the makers publish about their releases, read on 2026-11-18 without downloading anything
(the pre-build checks' C-L1: PyPI's JSON and each wheel's own PEP 658 metadata for LMDeploy 0.18.0, PyTorch's cu128
index for torch 2.11.0 and torchvision 0.26.0, uv 0.13.8's release files as plugin-platform-aider pins them). Five
files are fetched by this plugin itself, each checked against the sha256 pinned here before it is used: uv's archive
(from GitHub, which sends the file on to its release-assets host), the LMDeploy wheel (from PyPI's file host) and
PyTorch's torch and torchvision wheels for CUDA 12.8 (from PyTorch's own index). PyTorch's index lists no sizes, so
those two are checked by their sum alone. Everything else of the install uv takes as its indexes give it, over https:
that is said before the install, never claimed checked.

The layout is the base's one (machines/ai.py): the install of 0.18.0 is built in home/app/0.18.0, put in use by
home/app/CURRENT once it is whole, and every other folder of home/app/ is removed then:

    home/app/0.18.0/uv/bin/uv[.exe]   uv itself, unpacked from its archive
    home/app/0.18.0/uv/python/        the CPython 3.12.16 uv downloads for the environment (it stays with it)
    home/app/0.18.0/venv/             the environment: lmdeploy and hf (Scripts\\ on Windows, bin/ on Linux)
    home/downloads/                   the four files and uv's cache while an install runs (removed after a good one)

LMDeploy runs here with its TurboMind engine alone (the user, 2026-11-25: "TurboMind only, never the silent PyTorch
fallback"): its PyTorch engine always starts a second process holding an unauthenticated loopback port (C-L1 4b), and
on Windows it needs a triton LMDeploy does not install. Its wheels need no CUDA toolkit installed for the whole system:
on Windows TurboMind takes CUDA 12's libraries from PyTorch's own wheel, on Linux from NVIDIA's pip packages (C-L1 3d,
said there as likely and not read in the module's import table). The install ends with a windowless check that the
environment imports LMDeploy 0.18.0 and loads TurboMind's native module, so an install whose TurboMind would not load
is never put in use.

What a found install means: an `lmdeploy` on PATH that is not this plugin's (its user's own environment) is used as
it is to serve local instances, its version read from its environment's files alone, and is never updated, replaced
or removed from here."""
from __future__ import annotations

import math
import os
import platform
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from liminal_libs.machines import autostart, download, linux, uv
from liminal_libs.machines.ai import Installed, Package, Refusal, version_tuple
from liminal_libs.machines.plugin import CommandError
from liminal_libs.machines.world import size_words

# -- the pins (C-L1, read 2026-11-18) --------------------------------------------------------------------------------
VERSION = "0.18.0"  # PyPI's 0.18.0 of 2026-09-28 (GitHub's tag v0.18.0, commit 110965c)
PIN = VERSION  # the folder the install is built in: home/app/0.18.0
PYTHON = "3.12"  # cp312 wheels; at uv 0.13.8 it is CPython 3.12.16, build 20261110
PYTHON_SAID = "CPython 3.12.16"
TORCH = "2.11.0"  # the newest torch built for CUDA 12.8 within LMDeploy 0.18.0's torch<=2.12.1 (2.12 is cu130 only)
TORCHVISION = "0.26.0"  # its torchvision, within torchvision<=0.27.1
TORCH_BACKEND = "cu128"  # uv's --torch-backend: PyTorch's family from its CUDA 12.8 index, never --extra-index-url
# NVIDIA's driver CUDA 12.8 Update 1 needs (its release notes, Table 3), by system
DRIVERS = {"windows": "572.61", "linux": "570.124.06"}
UV = "0.13.8"
UV_RELEASE = f"https://github.com/astral-sh/uv/releases/download/{UV}/"
# system -> (uv's archive, its bytes, its sha256): uv's release JSON, its .sha256 files and its sha256.sum agree
UV_FILES = {
    "windows": ("uv-x86_64-pc-windows-msvc.zip", 15_529_884,
                "529e78a9cec6cd7e3e014f7ecd8bda0c113969d4ed0f2efeedb85c30943325b8"),
    "linux": ("uv-x86_64-unknown-linux-gnu.tar.gz", 16_855_069,
              "5acfd9177cfecf0bf87578f059728215ccc884388988f55f7a65041b57387d91"),
}
PYPI_FILES = "https://files.pythonhosted.org/packages/"
# system -> (LMDeploy's wheel, its bytes, its sha256, its address on PyPI's file host)
WHEELS = {
    "windows": ("lmdeploy-0.18.0-cp312-cp312-win_amd64.whl", 58_831_511,
                "26c974dcedf547d0e868998002bda691575932ae77c65e0cb9b027946610603e",
                PYPI_FILES + "77/d0/e2642c425be45809bb2059e77ae7aefdb503292cd9bd04d73b510552d320/"),
    "linux": ("lmdeploy-0.18.0-cp312-cp312-manylinux_2_28_x86_64.whl", 106_692_457,
              "fcef7cf0d210d766e02e2e6df025378d2f8ae36080400b2818eb9a34ef3392dc",
              PYPI_FILES + "19/8d/b880e02991ce03947c1efa9504c540d409b06d5d7ea4f95b1d8f8ae85c99/"),
}
TORCH_FILES = "https://download-r2.pytorch.org/whl/cu128/"  # where PyTorch's cu128 index sends its files
# system -> ((the wheel's name, its sha256), ...): torch, then torchvision; PyTorch's simple index lists no sizes
TORCH_WHEELS = {
    "windows": (("torch-2.11.0+cu128-cp312-cp312-win_amd64.whl",
                 "7c78215c3af4f62e63f2b2e360f1722fc719b0853c7ac22666483d9810613a4c"),
                ("torchvision-0.26.0+cu128-cp312-cp312-win_amd64.whl",
                 "8c0d1c4fbb2c9a4d5d41d0aaa87da20e525bcb2a154ce405725b0be59456804b")),
    "linux": (("torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl",
               "d252cf975fb18c94a85336323ad425f473df56dab35a44b00399bd70c7a3b997"),
              ("torchvision-0.26.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl",
               "ccf26b4b659cfce6f2208cb8326071d51c70219a34856dfdf468d1e19af52c0d")),
}
ASSET_HOSTS = ("release-assets.githubusercontent.com",)  # where a GitHub release file is sent on to (a 302)
HUB = "https://huggingface.co"  # the Hugging Face Hub: its API for a lookup (models.py), its pages for people
# every host the install reaches, said before it: uv's archive (GitHub and its asset host), uv's Python (Astral's
# mirror first, GitHub after a failure), LMDeploy's wheel and the packages (PyPI), PyTorch's cu128 index and its files
HOSTS = ("github.com", "release-assets.githubusercontent.com", "releases.astral.sh", "pypi.org",
         "files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org")

# -- sizes, room and time ----------------------------------------------------------------------------------------------
# what is downloaded, an estimate said as one until the first install measures it: PyTorch's CUDA wheel alone is
# about 2.5 GB, and on Linux NVIDIA's CUDA 12 runtime, cuDNN and NCCL come from PyPI besides (their sizes not read)
ABOUT = {"windows": 4 << 30, "linux": 7 << 30}
NEED_FREE = 20 << 30  # the free room asked for before an install: uv's cache and the environment side by side
TORCH_BYTES = 3 << 30  # torch's wheel, unpinned in size: the time its download is given is reckoned on this
PACKAGES_BYTES = 6 << 30  # what uv takes beside the four files, at the most (an estimate)
RATE = 250_000  # bytes a second: 2 Mbit/s, the slowest line a step is given the time for
MARGIN = 1.5  # what is not read is given half again
CHECK_S = 120.0  # the import check: torch's CUDA libraries load the first time for tens of seconds on Windows
CHECK = ("import sys, lmdeploy; from lmdeploy.turbomind import _turbomind; "
         "print('lmdeploy', lmdeploy.__version__, 'turbomind loads'); "
         f"sys.exit(0 if lmdeploy.__version__ == {VERSION!r} else 3)")


def seconds_for(size: int) -> float:
    """The time a step that downloads `size` bytes is given: at RATE, MARGIN over, in whole ten minutes, ten at
    least."""
    return float(max(600, math.ceil(size * MARGIN / RATE / 600) * 600))


VENV_S = seconds_for(40 << 20)  # uv's Python, unpacked
WHEEL_S = seconds_for(WHEELS["linux"][1])
TORCH_S = seconds_for(TORCH_BYTES)
PACKAGES_S = seconds_for(PACKAGES_BYTES)

# -- what every command of an install runs with ------------------------------------------------------------------------
# taken out of the platform's environment besides uv's own (uv.DROP) and Liminal's: nothing set elsewhere steers it
INSTALL_DROP = ("LMDEPLOY_*", "HF_*", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_*", "TRANSFORMERS_*", "MODELSCOPE_*",
                "TORCH_*", "TRITON_*", "RAY_*")
QUIET = {"DO_NOT_TRACK": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "RAY_USAGE_STATS_ENABLED": "0"}

LICENCE = ("Apache-2.0, LMDeploy's own (stated in its README; PyPI's licence field is empty). What it brings carries "
           "its own terms, which nothing here asks you to accept: PyTorch (BSD-3-Clause, with the licences of what "
           "its wheel holds), NVIDIA's CUDA libraries (NVIDIA's own proprietary terms), Ray (Apache-2.0), Hugging "
           "Face's libraries (Apache-2.0), uv (MIT or Apache-2.0) and its Python (the PSF's licence). Each model "
           "carries its own licence, said before it is downloaded.")
CHECKED = ("uv's archive, the LMDeploy wheel and PyTorch's torch and torchvision wheels are checked against the "
           "sha256 pinned in this plugin; what uv takes from PyPI and PyTorch's index besides them comes over https "
           "as those indexes give it, and no sum pinned here checks it.")
WSL1 = ("LMDeploy does not run under WSL 1, which has no Linux kernel and no GPU: make this distribution WSL 2 "
        "(`wsl --set-version <its name> 2`, from Windows)")
MUSL = ("LMDeploy's wheels are built for glibc (manylinux_2_28), and this Linux's C library is musl: it is not "
        "installed here")
FOUND_NOTE = (f"your own environment: it is used as it is, started with {VERSION}'s arguments, and is never updated, "
              "replaced or removed from here")
NO_GPU = ("LMDeploy's TurboMind needs an NVIDIA GPU: nvidia-smi reads none here ({why}). Instances elsewhere can still "
          "be added by their address")
TURBOMIND = ("TurboMind only: LMDeploy's PyTorch engine is never used here, since it always starts a second process "
             "that holds a loopback port with no key, and LMDeploy's own command falls back to it silently for a "
             "model TurboMind does not serve; such a model is refused before a start")
TURBOMIND_WINDOWS = ("TurboMind on Windows serves one GPU (no tensor parallelism), with kernels for the Volta, Turing, "
                     "Ampere (sm_86), Ada and Blackwell (sm_120) cards alone: an A100, H100 or B200 has none")


@dataclass(frozen=True)
class Here:
    """What this machine is, read from Python and from files alone (nothing run, nothing asked): a test's stand-ins.
    `machine`: its processor (platform.machine); `release`: its kernel's release (platform.release); `libc`: its C
    library and version (platform.libc_ver); `exists`: whether a path is there; `free`: the bytes free on the disk
    holding a folder; `systemd`: whether systemd is its init."""

    machine: Callable[[], str] = platform.machine
    release: Callable[[], str] = platform.release
    libc: Callable[[], tuple] = platform.libc_ver
    exists: Callable[[str], bool] = os.path.exists
    free: Callable[[Path], int] | None = None
    systemd: Callable[[], bool] = linux.systemd_runs

    def free_at(self, folder) -> int:
        """The bytes free on the disk that holds `folder` (its nearest parent that is there, when it is not yet)."""
        if self.free is not None:
            return int(self.free(Path(folder)))
        place = Path(folder)
        while not place.exists() and place != place.parent:
            place = place.parent
        return int(shutil.disk_usage(place).free)


# -- what is there -----------------------------------------------------------------------------------------------------
def windows(p) -> bool:
    return p.system == "windows"


def venv_in(folder: Path) -> Path:
    return Path(folder) / "venv"


def bin_of(p, venv: Path) -> Path:
    """An environment's programs: Scripts\\ on Windows, bin/ on Linux."""
    return Path(venv) / ("Scripts" if windows(p) else "bin")


def program_of(p, venv: Path, name: str) -> Path:
    return bin_of(p, venv) / (f"{name}.exe" if windows(p) else name)


def lmdeploy_in(p, folder: Path) -> Path:
    """The lmdeploy program of an install built in `folder`."""
    return program_of(p, venv_in(folder), "lmdeploy")


def uv_in(p, folder: Path) -> Path:
    """uv itself in an install's folder."""
    return uv.program_in(Path(folder) / "uv" / "bin", windows=windows(p))


def site_packages(p, venv: Path) -> Path:
    """Where an environment's packages are: Lib\\site-packages on Windows, lib/python3.12/site-packages on Linux."""
    return Path(venv) / ("Lib" if windows(p) else "lib") / ("" if windows(p) else f"python{PYTHON}") / "site-packages"


def dist_version(program) -> str:
    """The version of LMDeploy an environment holds, from its files alone: the lmdeploy-<version>.dist-info folder in
    its site-packages ("" when none is found). `program`: its lmdeploy (or a link to it), in Scripts\\ or bin/."""
    seen: set[str] = set()
    for path in (Path(program), Path(os.path.realpath(program))):
        root = path.parent.parent
        if str(root) in seen:
            continue
        seen.add(str(root))
        found: list[Path] = []
        for pattern in ("Lib/site-packages/lmdeploy-*.dist-info", "lib/python3*/site-packages/lmdeploy-*.dist-info"):
            try:
                found += sorted(root.glob(pattern))
            except OSError:
                continue
        for info in found:
            said = re.fullmatch(r"lmdeploy-(\d[0-9A-Za-z.+!-]*)\.dist-info", info.name)
            if said and info.is_dir():
                return said.group(1)
    return ""


def installed(p) -> Installed | None:
    """What is installed here, files looked at only: this plugin's install (the folder app/CURRENT names, its lmdeploy
    there); else, while none is in use, an lmdeploy of the user's own on PATH (found)."""
    folder = p.current()
    if folder is not None:
        program = lmdeploy_in(p, folder)
        return Installed(folder.name, str(program), "plugin") if program.is_file() else None
    found = p.out.which("lmdeploy")
    if not found:
        return None
    return Installed(dist_version(found), str(found), "found", FOUND_NOTE)


def program(p, have: Installed | None = None) -> tuple[str, ...]:
    """The head of a server's command line: the lmdeploy program installed (this plugin's, or the user's own); ()
    when none is."""
    have = p.installed() if have is None else have
    return (have.path,) if have is not None else ()


def hf_of(p, have: Installed) -> Path:
    """Hugging Face's hf command of an install's environment (huggingface_hub comes with LMDeploy's requirements):
    beside its lmdeploy; or, where that lmdeploy is a link into an environment of its own, beside the program it
    leads to. The first of the two when neither holds one."""
    name = "hf.exe" if windows(p) else "hf"
    beside = Path(have.path).parent / name
    if beside.is_file():
        return beside
    try:
        real = Path(os.path.realpath(have.path)).parent / name
    except (OSError, ValueError):
        return beside
    return real if real.is_file() else beside


# -- where it runs -----------------------------------------------------------------------------------------------------
def _machine(p) -> str:
    return str(p.here.machine() or "").strip().lower()


def _libc(p) -> tuple[str, str]:
    """This Linux's C library and its version, as Python reads them from its own program (a file read, nothing run),
    read once while the plugin lives: ("", "") when it cannot tell."""
    with p._rt.lock:
        kept = p._rt.libc
    if kept is None:
        try:
            libc, said = p.here.libc()
            kept = (str(libc or ""), str(said or ""))
        except (OSError, ValueError, TypeError):
            kept = ("", "")
        with p._rt.lock:
            p._rt.libc = kept
    return kept


def runs_here(p) -> str:
    """"" where LMDeploy installs and starts here (Windows on x64; Linux on x86-64 with glibc 2.28 or newer, a WSL 2
    distribution included), else why not, in words. Nothing is run to tell: Python's own reading of the machine, and a
    file looked for."""
    machine = _machine(p)
    if machine not in ("x86_64", "amd64"):
        return (f"LMDeploy's makers build its wheels for x86-64 alone, and this machine is {machine or 'not said'}: "
                "an instance elsewhere can be added by its address")
    if windows(p):
        return ""
    if p.wsl and linux.wsl_generation(str(p.here.release() or "")) == 1:
        return WSL1
    libc, said = _libc(p)
    if p.here.exists("/etc/alpine-release") or (libc and libc != "glibc"):
        return MUSL
    if libc == "glibc" and said and version_tuple(said) < (2, 28):
        return (f"LMDeploy's wheels need glibc 2.28 or newer (manylinux_2_28), and this Linux has glibc {said}: it is "
                "not installed here")
    return ""


def _files(p) -> str:
    return "windows" if windows(p) else "linux"


def package(p) -> Package | None:
    """What Install brings: on Windows x64 and Linux x86-64 (WSL 2 included) LMDeploy 0.18.0 and what it needs;
    elsewhere none."""
    if _machine(p) not in ("x86_64", "amd64"):
        return None
    which = _files(p)
    uv_file, wheel = UV_FILES[which][0], WHEELS[which]
    parts = (f"uv {UV} ({uv_file}, from GitHub) and a {PYTHON_SAID} of its own, then LMDeploy {VERSION} ({wheel[0]}, "
             f"{size_words(wheel[1])}, from PyPI) with PyTorch {TORCH} and torchvision {TORCHVISION} built for CUDA "
             "12.8 (from PyTorch's own index; torch alone is about 2.5 GB)"
             + (", NVIDIA's CUDA 12 runtime, cuDNN and NCCL from PyPI" if which == "linux" else "")
             + f" and the packages LMDeploy needs, as wheels alone (nothing is built). The size said is an estimate; "
             f"{size_words(NEED_FREE)} free is asked for, since uv's cache lies beside the environment until the "
             "install is whole")
    keeps = (f"Its caches ({p.home / 'cache'}) are deleted with it; its models ({p.home / 'models'}), its logs and "
             "your instances stay.")
    return Package("LMDeploy", VERSION, ABOUT[which], LICENCE, HOSTS, parts=parts, keeps=keeps, pin=PIN,
                   checked=CHECKED)


def uv_source(p) -> download.Source:
    name, size, sha = UV_FILES[_files(p)]
    return download.Source(UV_RELEASE + name, name, size, sha, ASSET_HOSTS)


def wheel_source(p) -> download.Source:
    name, size, sha, where = WHEELS[_files(p)]
    return download.Source(where + name, name, size, sha)


def torch_sources(p) -> tuple[download.Source, ...]:
    """torch's and torchvision's wheels for CUDA 12.8: checked by their sha256 alone (PyTorch's index lists no
    sizes)."""
    return tuple(download.Source(TORCH_FILES + name.replace("+", "%2B"), name, 0, sha)
                 for name, sha in TORCH_WHEELS[_files(p)])


# -- the install -------------------------------------------------------------------------------------------------------
def environment(p, folder: Path) -> dict[str, str]:
    """The whole environment every command of an install is run with: the platform's without uv's own variables,
    pip's, Python's, LMDeploy's, the Hub's, PyTorch's, Triton's, Ray's and every pair of Liminal's (uv.child_env), uv's
    own settings over it (everything it keeps under `folder`, its cache in home/downloads/uv), and the Hub's telemetry
    and Ray's usage statistics off."""
    env = uv.child_env(folder, base=p.out.env, drop=(*p.dropped(), *INSTALL_DROP), system=p.system,
                       cache=p.home / "downloads" / "uv")
    env.update(QUIET)
    return env


def driver_tuple(text) -> tuple:
    """A driver's version as numbers (572.61 is (572, 61)); () for none."""
    said = re.match(r"\s*(\d+(?:\.\d+)*)", str(text or ""))
    return tuple(int(n) for n in said.group(1).split(".")) if said else ()


def _gpu_checked(p, say) -> None:
    """The install's first refusals that need nvidia-smi (a job may run it, a status never): no NVIDIA card read, or
    a driver older than CUDA 12.8 needs on this system."""
    gpus = p.gpus()
    if not gpus.cards:
        raise Refusal(NO_GPU.format(why=gpus.why or "no answer"))
    card = max(gpus.cards, key=lambda g: (g.total, g.free))
    needed = DRIVERS[_files(p)]
    if driver_tuple(card.driver) < driver_tuple(needed):
        where = " (it is Windows' driver: update it there)" if p.wsl else ""
        raise Refusal(f"LMDeploy {VERSION}'s CUDA 12.8 wheels need NVIDIA's driver {needed} or newer here; this one is "
                      f"{card.driver or 'not said'}{where}")
    say(f"the GPU: {card.name}, {size_words(card.total)}, driver {card.driver}: CUDA 12.8 needs {needed} or newer")


def _room(p) -> None:
    free = p.here.free_at(p.home)
    if free < NEED_FREE:
        raise Refusal(f"LMDeploy's install needs about {size_words(NEED_FREE)} free in {p.home}, and "
                      f"{size_words(free)} is free there: at its peak uv's cache and the environment lie side by side, "
                      "beside the files fetched for it")


def _forget(path: Path, inside: Path) -> list[str]:
    """A download no longer needed once the install is whole, removed (a file or uv's cache); its line."""
    if not os.path.lexists(path):
        return []
    try:
        freed = download.remove_tree(path, inside)
    except (OSError, ValueError) as e:
        return [f"{path} could not be removed ({e}): the next uninstall removes it"]
    return [f"removed {path.name} from downloads/ ({size_words(freed)})"]


def install_into(p, log, cancel) -> list[str]:
    """LMDeploy installed into its own folder, as a job: the GPU, its driver and the room looked at first (refused in
    words), uv's archive fetched and unpacked, the environment made on uv's own Python, the LMDeploy, torch and
    torchvision wheels fetched and checked, then given to uv with PyTorch's CUDA 12.8 index for the rest of PyTorch's
    family and wheels alone (--no-build), then a windowless check that it imports LMDeploy 0.18.0 and loads TurboMind;
    the install put in use only then. Its lines for the answer."""
    say = log
    _gpu_checked(p, say)
    _room(p)
    folder = p.building(PIN)
    downloads = p.home / "downloads"
    source = uv_source(p)
    archive = p.fetch(source, downloads, log=log, cancel=cancel)
    p.unpack(archive, folder / "uv" / "bin", log=log, cancel=cancel, strip=uv.strip_of(source.name))
    program_uv = uv_in(p, folder)
    if not program_uv.is_file():
        raise download.DownloadError(f"{source.name} holds no uv: nothing was installed")
    env = environment(p, folder)
    runner = p.stream_whole
    venv = uv.venv(program_uv, venv_in(folder), PYTHON, env=env, log=log, cancel=cancel, timeout=VENV_S, stream=runner)
    wheel = p.fetch(wheel_source(p), downloads, log=log, cancel=cancel, timeout=WHEEL_S)
    torches = [p.fetch(s, downloads, log=log, cancel=cancel, timeout=TORCH_S) for s in torch_sources(p)]
    say(f"{wheel.name}, {torches[0].name} and {torches[1].name} are checked against the sha256 pinned in this plugin. "
        "uv's line below, \"no sum pinned here checks it\", is true of what uv takes for them from PyPI and PyTorch's "
        "index: those come over https as the indexes give them")
    uv.pip_install(program_uv, venv, [str(wheel), *(str(t) for t in torches)], env=env, log=log, cancel=cancel,
                   extra=("--no-build", f"--torch-backend={TORCH_BACKEND}"), timeout=PACKAGES_S, stream=runner)
    info = site_packages(p, venv) / f"lmdeploy-{VERSION}.dist-info"
    if not info.is_dir() or not lmdeploy_in(p, folder).is_file():
        raise download.DownloadError(f"LMDeploy {VERSION} is not found in {venv} after uv's install: nothing is put in "
                                     "use")
    say(f"the check: the environment's Python imports LMDeploy {VERSION} and loads TurboMind's native module")
    code = p.run_whole([str(program_of(p, venv, "python")), "-I", "-c", CHECK], log, env=env, timeout=CHECK_S,
                       cancel=cancel, cwd=str(folder))
    if code != 0:
        raise CommandError(f"the environment does not load LMDeploy {VERSION}'s TurboMind (the check ended with "
                           f"{code}; its last lines are in the panel above): nothing is put in use, since LMDeploy "
                           "would fall back to its PyTorch engine")
    lines = [*p.adopt(PIN), f"LMDeploy {VERSION} in {venv}, its TurboMind loading"]
    for done in (downloads / "uv", wheel, *torches, archive):
        lines += _forget(done, p.home)
    lines.append(CHECKED)
    return lines


def remove_from(p) -> list[str]:
    """The uninstall's work: the install and what was downloaded for it (the base's), then its caches in home/cache;
    its models, its logs and the instances stay."""
    lines = [*p.forget_install(), *p.forget_downloads()]
    cache = p.home / "cache"
    if os.path.lexists(cache):
        try:
            lines.append(f"removed its caches, {cache} ({size_words(download.remove_tree(cache, p.home))})")
        except (OSError, ValueError) as e:
            lines.append(f"{cache} could not be removed whole ({e}): what is left is removed by the next uninstall")
    lines.append(f"kept: its models in {p.home / 'models'}, its logs, your instances")
    return lines


# -- what this system lacks ------------------------------------------------------------------------------------------
TITLE = "On this system"
COLUMNS = [{"title": "What"}, {"title": "Here"}, {"title": "What it means"}]


def _gpu_command(p) -> str:
    """Where nvidia-smi is, looked for and never run (gpu.command with the plugin's own lookups)."""
    from liminal_libs.machines import gpu

    try:
        return gpu.command(p.out.which, p.here.exists)
    except (OSError, ValueError, TypeError):
        return ""


def lacks(p) -> dict:
    """The table "On this system": what Windows, this Linux or WSL changes for LMDeploy, from files and kept readings
    alone (nothing run: nvidia-smi is only looked for, and read on the Models tab and inside the install)."""
    rows: list[list[str]] = []
    here = runs_here(p)
    if here:
        rows.append(["LMDeploy", "not here", here])
    needed = DRIVERS[_files(p)]
    found = _gpu_command(p)
    if found:
        rows.append(["GPU", f"nvidia-smi at {found}",
                     "its card and its driver are read on the Models tab and when LMDeploy is installed"])
    else:
        rows.append(["GPU", "nvidia-smi is not on this system", NO_GPU.format(why="nvidia-smi is not there")])
    rows.append(["Driver", f"{needed} or newer", "what CUDA 12.8, which LMDeploy's and PyTorch's wheels are built for, "
                                                 "needs on this system" + (" (Windows' driver, under WSL)"
                                                                           if p.wsl else "")])
    rows.append(["Engine", "TurboMind only", TURBOMIND])
    if windows(p):
        rows.append(["TurboMind on Windows", "one GPU", TURBOMIND_WINDOWS])
    if p.wsl:
        rows.append(["This Linux", f"WSL 2: {p.wsl}",
                     "the GPU is Windows' NVIDIA driver's, passed through: none is installed inside. Windows reaches a "
                     "server here at its own 127.0.0.1, where Windows' LMDeploy plugin lists it as found."])
        rows.append(["Memory (WSL)", "half of the PC's by default",
                     "WSL gives Linux half of the PC's memory unless .wslconfig says more, and a model's files pass "
                     "through it while they load"])
    if windows(p):
        rows.append(["Start at boot", "a scheduled task", autostart.how("windows")])
    else:
        why = autostart.why_not("linux", systemd=p.systemd_runs(), wsl=p.wsl)
        kept = p.lasting()
        if why:
            rows.append(["Start at boot", "not here", why])
        elif kept:
            rows.append(["Start at boot", "refused for now", f"{kept} Start at boot needs its instances kept across a "
                                                             "restart."])
        else:
            rows.append(["Start at boot", "a systemd user unit", autostart.how("linux", p.wsl)])
    return {"title": TITLE, "columns": COLUMNS, "rows": rows}


# the calls this module makes, by card (apis.py's names): tests/test_apis.py holds the cards to them
CALLS = {
    "uv": ("GET uv's archive", "uv venv", "uv pip install"),
    "lmdeploy, the command": ("python -I -c (the import check)",),
    "The package indexes": ("GET LMDeploy's wheel", "GET torch's and torchvision's wheels"),
    "nvidia-smi": ("nvidia-smi --query-gpu",),
}
