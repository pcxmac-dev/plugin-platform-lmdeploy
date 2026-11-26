"""What every test here shares: the guard, and the stand-ins the plugin is driven with.

Nothing real is reached. The guard (`nothing_outside`, for every test) fails a test whose code starts a process, opens
a connection, listens for one or resolves a name: the stand-ins raise `Reached`, a BaseException (no `except
Exception` swallows it), and every reach is noted and failed at the test's end too. No LMDeploy, uv, hf, Python,
nvidia-smi, systemctl or PowerShell runs; no file is downloaded; no port is opened.

The machine the tests run on never shows through: the plugin is given an Outside of stand-ins (its environment, its
system, its WSL distribution, its folder, its clock, its lookups, its socket table, its http, its downloads, its
commands, its short commands) and a Here of stand-ins (its processor, its kernel's release, its C library, the files
looked for, the free room, systemd), so the suite reads alike on Windows, inside WSL and on a Linux of its own. The
library's own judgement of another system's drive answers "not foreign" unless a test says otherwise, and the boot
registrations are a stand-in's. The recorded answers in tests/data are written from LMDeploy's and the Hub's own
words (C-L1 read them in LMDeploy 0.18.0's sources), not recorded from a run. After each test no thread of the
plugin's is left."""
from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import subprocess
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import pytest

try:
    import requests  # noqa: F401 - imported before the guard: urllib3 binds a socket of its own when first imported
except ImportError:  # (not every run has it)
    pass

from liminal_libs import mounts
from liminal_libs.apps.authorization import Login, Unauthorized, pair_name, variable_name
from liminal_libs.machines import autostart
from liminal_libs.machines.admin import Startup
from liminal_libs.machines.ai import Outside
from liminal_libs.machines.endpoint import Reply
from liminal_libs.machines.terminal import Cancelled
from liminal_libs.resources import Conn

from plugin_platform_lmdeploy import Here, LmdeployPlugin, release

DATA = Path(__file__).resolve().parent / "data"
NOW = 1_763_400_000.0
GPU_LINE = (DATA / "nvidia-smi.txt").read_text(encoding="utf-8")  # its answer to the query gpu.py runs
TOKEN = "hf_StandInTokenOfTheTests0123456789"  # a token a test keeps: it is in no answer, line or error
DECLARED = ("MASSIVE_API_KEY", "LIMINAL_AI_LMDEPLOY_HF_TOKEN")  # what the registry says the installed plugins declare
WSL_RELEASE = "6.6.87.2-microsoft-standard-WSL2"
PREFIXES = ("LMDEPLOY_", "HF_", "LIMINAL_AI_", "UV_", "TIKTOKEN_", "RAY_")

# -- the guard -----------------------------------------------------------------------------------------------------
STOOD_IN = [
    (subprocess, "subprocess", ("Popen", "run", "call", "check_call", "check_output", "getoutput", "getstatusoutput",
                                "_fork_exec")),
    (os, "os", ("system", "startfile", "posix_spawn", "posix_spawnp", "spawnv", "spawnve", "spawnvp", "spawnvpe",
                "fork", "forkpty", "execv", "execve")),
    (socket, "socket", ("create_connection", "create_server", "socketpair", "getaddrinfo", "gethostbyname",
                        "gethostbyname_ex", "gethostbyaddr", "getnameinfo")),
    (socket.socket, "socket.socket", ("connect", "connect_ex", "bind", "listen", "sendto", "sendmsg")),
]
if os.name == "nt":
    import _winapi

    STOOD_IN.append((_winapi, "_winapi", ("CreateProcess",)))
else:
    import _posixsubprocess

    STOOD_IN.append((_posixsubprocess, "_posixsubprocess", ("fork_exec",)))


class Reached(BaseException):
    """A test reached for real what it must stand in for."""


@pytest.fixture(autouse=True)
def nothing_outside(monkeypatch):
    reached: list[str] = []

    def refuse(what):
        def stand_in(*args, **kwargs):
            __tracebackhide__ = True  # (its arguments, a process's environment with its keys, never printed)
            reached.append(what)
            raise Reached(f"the test reached {what} for real: give the code a stand-in")
        return stand_in

    for owner, said, names in STOOD_IN:
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, refuse(f"{said}.{name}"))
    monkeypatch.setattr(mounts, "foreign", lambda path, *a, **k: False)
    for name in list(os.environ):
        if name.upper().startswith(PREFIXES):
            monkeypatch.delenv(name, raising=False)
    yield reached
    assert not reached, f"the test reached the outside for real: {reached}"


@pytest.fixture(autouse=True)
def no_thread_left():
    yield
    deadline = time.monotonic() + 5.0
    for t in threading.enumerate():
        if t.name.startswith(("boot-lmdeploy", "hub-download-progress")):
            t.join(max(0.0, deadline - time.monotonic()))
    left = [t.name for t in threading.enumerate()
            if t.name.startswith(("boot-lmdeploy", "hub-download-progress")) and t.is_alive()]
    assert not left, f"threads of the plugin's left after the test: {left}"


# -- the stand-ins ---------------------------------------------------------------------------------------------------
class Pairs(Login):
    """The platform's login as a test has it: pairs kept in memory, each save noted."""

    def __init__(self):
        self.kept: dict[str, str] = {}
        self.saved: list[tuple[str, bool]] = []

    def user(self) -> dict:
        return {"name": "someone"}

    def vector(self, name: str) -> str:
        variable = variable_name(name)
        if variable not in self.kept:
            raise Unauthorized(f"no {name}", 404)
        return self.kept[variable]

    def names(self) -> set[str]:
        return {pair_name(v) for v in self.kept}

    def save(self, variable: str, value: str | None) -> str:
        self.saved.append((variable, value is not None))
        if value is None:
            self.kept.pop(variable, None)
        else:
            self.kept[variable] = value
        return "the test's pairs"


class Clock:
    def __init__(self, now: float = NOW):
        self.now = now

    def __call__(self) -> float:
        return self.now


class Probe:
    """This system's socket table as a test sets it."""

    def __init__(self):
        self.conns: list[Conn] = []
        self.names: dict[int, str] = {}

    def listen(self, pid: int, port: int, name: str = "", address: str = "127.0.0.1") -> None:
        self.conns.append(Conn(pid, "listening", f"{address}:{port}", ""))
        if name:
            self.names[pid] = name

    def connections(self):
        return list(self.conns)

    def program_names(self, pids=None):
        return {p: n for p, n in self.names.items() if pids is None or p in pids}

    def processes(self):
        return []


class Transport:
    """http as a test answers it: by method and path, each ask noted with its url and headers. An answer is
    (status, JSON or bytes, headers)."""

    def __init__(self):
        self.asked: list[dict] = []
        self.answers: dict[tuple[str, str], tuple] = {}
        self.silent: set[tuple[str, str]] = set()  # asks that get no answer at all (nothing listens)

    def answer(self, method: str, path: str, status: int, data=None, headers=None) -> None:
        self.answers[(method, path)] = (status, data, headers or {})

    def __call__(self, method, url, headers, body, timeout, stream):
        path = urlsplit(url).path
        self.asked.append({"method": method, "url": url, "path": path, "headers": dict(headers), "body": body})
        if (method, path) in self.silent:
            raise ConnectionRefusedError("[Errno 111] Connection refused")
        status, data, more = self.answers.get((method, path), (404, {"error": "not here"}, {}))
        raw = data if isinstance(data, bytes) else json.dumps(data).encode()
        return Reply(status, {"content-type": "application/json", **more}, [raw])

    def hosts(self) -> set[str]:
        return {urlsplit(a["url"]).hostname for a in self.asked}


class Opener:
    """A download's http as a test answers it: a file's bytes by its address, each ask noted."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.asked: list[tuple[str, dict]] = []

    def __call__(self, url, headers, timeout):
        self.asked.append((url, dict(headers)))
        body = self.files.get(url)
        if body is None:
            return Reply(404, {}, [b"not here"])
        return Reply(200, {"content-length": str(len(body))}, [body[i:i + 65536] for i in range(0, len(body), 65536)])


class Streamer:
    """The commands a plugin runs to their end (uv, the import check, hf), as a test answers them: each run noted
    (its words, its whole environment, its folder, its time), its lines handed on, what it makes made by the test's
    `effects`, its exit code; `cancel_at`: the command whose run its user cancels."""

    def __init__(self):
        self.ran: list[dict] = []
        self.codes: dict[str, int] = {}
        self.lines: dict[str, list[str]] = {}
        self.effects: dict = {}
        self.cancel_at = ""

    @staticmethod
    def key(argv) -> str:
        name = Path(str(argv[0])).name.lower().removesuffix(".exe")
        return f"{name} {argv[1]}" if len(argv) > 1 else name

    def __call__(self, argv, line, *, env, cwd=None, timeout=None, cancel=None):
        key = self.key(argv)
        self.ran.append({"key": key, "argv": [str(a) for a in argv], "env": dict(env), "cwd": cwd, "timeout": timeout,
                         "cancel": cancel})
        for text in self.lines.get(key, ()):
            line(text)
        effect = self.effects.get(key)
        if effect is not None:
            effect(list(argv), dict(env))
        if self.cancel_at == key:
            raise Cancelled(f"{key} was cancelled: ended with what it started")
        return self.codes.get(key, 0)

    def keys(self) -> list[str]:
        return [r["key"] for r in self.ran]

    def run_of(self, key: str) -> dict:
        return next(r for r in self.ran if r["key"] == key)


class Kids:
    """A stand-in children.Child class, its own for each test."""

    def __init__(self, clock: Clock):
        made: list = []
        adoptable: dict = {}
        self.made, self.adoptable = made, adoptable

        class Child:
            def __init__(self, name, launch, log, pidfile, *, popen=None, interrupt=None, clock=None, sleep=None,
                         system=None):
                self.name, self.launch, self.log, self.pidfile = name, launch, Path(log), Path(pidfile)
                self.alive, self.pid, self.code, self.since, self.note = False, None, None, None, ""
                self.stops = 0
                made.append(self)

            def start(self):
                self.alive, self.pid, self.since = True, 4300 + len(made), clock()
                self.pidfile.parent.mkdir(parents=True, exist_ok=True)
                self.pidfile.write_text(f"{self.pid} 1 99 1", encoding="ascii")
                return self.pid

            @classmethod
            def adopt(cls, name, log, pidfile, **kw):
                found = adoptable.get(str(Path(pidfile)))
                return found if found is not None and found.alive else None

            def owns(self, pid):
                return pid == self.pid

            def stop(self, grace=8.0):
                self.stops += 1
                self.alive, self.code = False, 0
                return "ended when asked, after 0.2 s"

            def wait(self, timeout=None):
                return self.code

            def tail(self, lines=40):
                return ["its last line"][-lines:]

        self.Child = Child


class Boot:
    """autostart as a test has it: each call noted, the registrations kept here."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.kept: dict[str, Startup] = {}

    def register(self, plugin, instance, command, about, *, system, powershell=None, shell=None, home=None):
        self.calls.append(("register", plugin, instance, system))
        found = Startup(instance, autostart.name_of(plugin, instance, system), "ready", " ".join(command))
        self.kept[instance] = found
        return found

    def remove(self, plugin, instance, *, system, powershell=None, shell=None, home=None):
        self.calls.append(("remove", plugin, instance, system))
        return self.kept.pop(instance, None) is not None

    def act(self, plugin, instance, action, *, system, powershell=None, shell=None):
        self.calls.append(("act", plugin, instance, action, system))

    def read(self, plugin, *, system, powershell=None, shell=None, home=None):
        self.calls.append(("read", plugin, system))
        return dict(self.kept)


class App:
    """The platform as a plugin reaches it."""

    def __init__(self, *plugins, once=False):
        self.by = {p.name: p for p in plugins}
        self.once = once
        self.asked: list[tuple[str, str]] = []

    def plugins(self):
        return {n: p.interface() for n, p in self.by.items()}

    def ask(self, name, hook, args):
        self.asked.append((name, hook))
        return self.by[name].ask(hook, *args)


class Panel:
    """A job's panel as the platform hands one (its Output): its lines, and a cancel."""

    def __init__(self):
        self.lines: list[str] = []
        self._cancelled = False

    def __call__(self, line: str) -> None:
        self.lines.append(str(line))

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def wait_cancelled(self, timeout: float) -> bool:
        return self._cancelled


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def uv_archive(system: str) -> bytes:
    """A stand-in of uv's archive: Windows' .zip holding uv.exe at its root, Linux's .tar.gz one folder holding uv
    and uvx, as its maker's workflow packs them."""
    out = io.BytesIO()
    body = b"stand-in uv\n"
    if system == "windows":
        with zipfile.ZipFile(out, "w") as packed:
            for name in ("uv.exe", "uvx.exe"):
                packed.writestr(name, body)
        return out.getvalue()
    with tarfile.open(fileobj=out, mode="w:gz") as packed:
        for name in ("uv", "uvx"):
            info = tarfile.TarInfo(f"uv-x86_64-unknown-linux-gnu/{name}")
            info.size, info.mode = len(body), 0o755
            packed.addfile(info, io.BytesIO(body))
    return out.getvalue()


def data(name: str):
    text = (DATA / name).read_text(encoding="utf-8")
    return json.loads(text) if name.endswith(".json") else text


@pytest.fixture
def boot(monkeypatch):
    b = Boot()
    for name in ("register", "remove", "act", "read"):
        monkeypatch.setattr(autostart, name, getattr(b, name))
    return b


class World:
    """Everything outside the plugin, and the plugin made with it (`plugin`)."""


def pin_stand_ins(monkeypatch, w) -> None:
    """The pins of the files this plugin fetches itself, made the stand-ins' (their real sums are no test's): uv's
    archive and LMDeploy's wheel by size and sum, torch's and torchvision's by their sum alone."""
    files = dict(release.UV_FILES)
    wheels = dict(release.WHEELS)
    torches = dict(release.TORCH_WHEELS)
    w.bytes = {}
    for system in ("windows", "linux"):
        archive = uv_archive(system)
        name = files[system][0]
        files[system] = (name, len(archive), sha(archive))
        w.opener.files[release.UV_RELEASE + name] = archive
        w.bytes[name] = archive
        wheel = f"PK stand-in lmdeploy wheel for {system}".encode()
        name, _, _, where = wheels[system]
        wheels[system] = (name, len(wheel), sha(wheel), where)
        w.opener.files[where + name] = wheel
        w.bytes[name] = wheel
        pairs = []
        for name, _ in torches[system]:
            body = f"PK stand-in {name}".encode()
            pairs.append((name, sha(body)))
            w.opener.files[release.TORCH_FILES + name.replace("+", "%2B")] = body
            w.bytes[name] = body
        torches[system] = tuple(pairs)
    monkeypatch.setattr(release, "UV_FILES", files)
    monkeypatch.setattr(release, "WHEELS", wheels)
    monkeypatch.setattr(release, "TORCH_WHEELS", torches)


@pytest.fixture
def world(tmp_path, boot, monkeypatch):
    w = World()
    w.tmp, w.boot = tmp_path, boot
    w.clock, w.probe, w.transport, w.opener, w.streamer = Clock(), Probe(), Transport(), Opener(), Streamer()
    w.kids = Kids(w.clock)
    w.ran: list[list[str]] = []
    w.gpu = GPU_LINE
    w.found: dict[str, str] = {"nvidia-smi": "/usr/bin/nvidia-smi"}
    w.free = 500 << 30
    w.files: set[str] = set()

    def runner(args, timeout):
        w.ran.append(list(args))
        return (0, w.gpu, "") if w.gpu is not None else (9, "", "NVIDIA-SMI has failed")

    def which(name, *a, **k):
        return w.found.get(name)

    def listen(host, port):
        return False

    w.runner, w.which = runner, which
    pin_stand_ins(monkeypatch, w)

    def env_of(system):
        return {"PATH": "/usr/local/bin:/usr/bin" if system == "linux" else "C:/Windows", "HOME": "/home/someone",
                "MASSIVE_API_KEY": "m-key-1", "LIMINAL_AI_LMDEPLOY_INSTANCES": "kept elsewhere",
                "LMDEPLOY_USE_MODELSCOPE": "true", "LMDEPLOY_EXECUTOR_BACKEND": "ray", "LMDEPLOY_LOG_FILE": "/tmp/x",
                "HF_TOKEN": "hf_someone_elses_token_000", "HF_HUB_OFFLINE": "0",
                "UV_INDEX_URL": "https://index.example.test/simple", "PIP_INDEX_URL": "https://index.example.test/pip",
                "PYTHONPATH": "/home/someone/lib", "HUGGING_FACE_HUB_TOKEN": "hf_an_old_token_name_000",
                "TRANSFORMERS_CACHE": "/elsewhere", "MODELSCOPE_CACHE": "/elsewhere", "TORCH_HOME": "/elsewhere",
                "TRITON_CACHE_DIR": "/elsewhere", "RAY_ADDRESS": "10.0.0.9:6379", "SSL_KEYFILE": "/k.pem",
                "UVICORN_LOG_LEVEL": "debug", "TIKTOKEN_ENCODINGS_BASE": "https://elsewhere.example.test/",
                "LIMINAL_PLATFORM_DIR": str(tmp_path / "platform")}

    def plugin(system="linux", wsl="Ubuntu", machine="x86_64", release_="", libc=("glibc", "2.39"), attach=True,
               systemd=True, login=True, **more):
        here = Here(machine=lambda: machine, release=lambda: release_ or (WSL_RELEASE if wsl else "6.8.0-generic"),
                    libc=lambda: libc, exists=lambda path: str(path) in w.files, free=lambda path: w.free,
                    systemd=lambda: systemd)
        given = dict(env=env_of(system), system=system, wsl=wsl if system == "linux" else "",
                     home=tmp_path / "platform" / "ai" / "lmdeploy", clock=w.clock, which=which, probe=w.probe,
                     transport=w.transport, opener=w.opener, streamer=w.streamer, runner=runner, listening=listen,
                     variables=lambda: DECLARED)
        given.update(more)
        p = LmdeployPlugin(Outside(**given), here)
        p.CHILD = w.kids.Child
        if login:
            p.login = Pairs()
        if attach:
            p.app = App(p)
        return p

    w.plugin = plugin
    return w


# -- what a test makes on the stand-in disk ----------------------------------------------------------------------------
def venv_bin(p, folder: Path) -> Path:
    return folder / "venv" / ("Scripts" if p.system == "windows" else "bin")


def make_install(p, version: str = release.VERSION) -> Path:
    """This plugin's install as a whole one is laid out: app/<pin>/venv's lmdeploy, hf and python, its dist-info,
    CURRENT."""
    folder = p.home / "app" / version
    bin_ = venv_bin(p, folder)
    bin_.mkdir(parents=True)
    end = ".exe" if p.system == "windows" else ""
    for name in ("lmdeploy", "hf", "python"):
        (bin_ / f"{name}{end}").write_text("stand-in\n", encoding="utf-8")
    (release.site_packages(p, folder / "venv") / f"lmdeploy-{version}.dist-info").mkdir(parents=True)
    (p.home / "app" / "CURRENT").write_text(version + "\n", encoding="utf-8")
    return folder


SNAPSHOT = "c1899de289a04d12100db370d81485cdf75e47ca"


def make_model(p, repo: str, files: dict[str, int] | None = None, incomplete: bool = False,
               config: dict | None = None) -> Path:
    """A repository downloaded into the plugin's models folder, in Hugging Face's layout (blobs, a snapshot whose
    files are links to them where this system lets a link be made, else copies, refs/main), its config.json
    `config` (Qwen3's by default; None written as it is: {} for none)."""
    folder = p.home / "models" / "hub" / ("models--" + repo.replace("/", "--"))
    blobs, snap = folder / "blobs", folder / "snapshots" / SNAPSHOT
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (folder / "refs").mkdir()
    (folder / "refs" / "main").write_text(snap.name, encoding="utf-8")
    for i, (name, size) in enumerate((files or {"model.safetensors": 4096}).items()):
        blob = blobs / f"{i:040x}"
        blob.write_bytes(b"x" * size)
        try:
            os.symlink(os.path.join("..", "..", "blobs", blob.name), snap / name)
        except (OSError, NotImplementedError):
            (snap / name).write_bytes(b"")  # (no link here: an empty stand-in; the blob holds the bytes)
    said = data("qwen3-config.json") if config is None else config
    if said:
        (snap / "config.json").write_text(json.dumps(said), encoding="utf-8")
    if incomplete:
        (blobs / ("f" * 40 + ".incomplete")).write_bytes(b"y" * 100)
    return folder


def install_effects(p, streamer: Streamer) -> None:
    """What uv's two commands make, as they would: the environment (uv venv) and LMDeploy in it (uv pip install)."""
    end = ".exe" if p.system == "windows" else ""

    def venv(argv, env):
        target = Path(argv[-1])
        bin_ = target / ("Scripts" if p.system == "windows" else "bin")
        bin_.mkdir(parents=True, exist_ok=True)
        (bin_ / f"python{end}").write_text("", encoding="utf-8")
        (target / "pyvenv.cfg").write_text(f"uv = {release.UV}\n", encoding="utf-8")

    def pip(argv, env):
        python = Path(argv[argv.index("--python") + 1])
        root = python.parent.parent
        bin_ = root / ("Scripts" if p.system == "windows" else "bin")
        bin_.mkdir(parents=True, exist_ok=True)
        for name in ("lmdeploy", "hf"):
            (bin_ / f"{name}{end}").write_text("stand-in\n", encoding="utf-8")
        (release.site_packages(p, root) / f"lmdeploy-{release.VERSION}.dist-info").mkdir(parents=True, exist_ok=True)

    streamer.effects["uv venv"] = venv
    streamer.effects["uv pip"] = pip
