"""LMDeploy for Liminal's platform: a model server plugin (liminal-libs 1.66.0's ModelServer), drawn under the
platform's AI heading in the Inference group, its icon LMDeploy's own mark (the bolt, drawn anew in the set's line).

The user's words (2026-11-18): "top 4 inference engines, top 4 agentic harnesses (open source)", "both linux and
windows compatibility"; and (2026-11-25, the fourth engine): LMDeploy, TurboMind only, never the silent PyTorch
fallback; the ports a tool opens by itself said on its page. Before them (2026-11-16): "all three plugins need to
allow installation and configuration of their respective code"; "also remote connection to instances if possible";
"the plugins manage the services, the app can be started @ boot like a service, ports can be configured via the
plugin, the plugin should report conflicts". And (2026-11-17): "all plugins are middleware", "even for other
plugins", "and then the app object code".

What it is, as middleware: a ModelServer, so it answers the standard interface every inference plugin answers (its
instances, their starts and stops, their ports and conflicts, their start at boot, their models, a prompt to try,
`serving` and `models_of` for the harnesses, asked through the platform), and adds four hooks of its own for its
models (`find`, `clear_found`, `pull`, `delete_model`). The software under it is LMDeploy's api_server and its command,
uv, Hugging Face's hf and Hub, nvidia-smi, and the scheduled tasks and systemd user units a start at boot is (apis.py:
its External API).

- **Where it runs.** Windows x64 and Linux x86-64 with glibc 2.28 or newer, a WSL 2 distribution included: LMDeploy's
  wheels are native on both. Elsewhere it holds instances by their address alone.
- **Install** (release.py): uv 0.13.8 from its pinned archive, an environment on uv's own CPython 3.12.16, LMDeploy
  0.18.0's wheel and PyTorch 2.11.0's CUDA 12.8 torch and torchvision wheels fetched and checked against their pinned
  sha256, then given to uv with PyTorch's cu128 index for the rest of its family, as wheels alone; then a windowless
  check that TurboMind loads. All inside home/app/0.18.0, put in use by home/app/CURRENT once whole. Its size (an
  estimate), its hosts and its licence are said first; the licence is accepted in a second confirmation; Cancel stops
  it. Nothing asks for an administrator, and nothing runs sudo. An lmdeploy of the user's own on PATH is used as it is,
  never replaced.
- **A server** (server.py): one model each, started offline as a windowless child this plugin owns (or by its
  scheduled task or systemd user unit, when it starts at boot), by TurboMind alone (a model it does not serve is
  refused before the start, and a server whose log says it fell back to PyTorch is stopped), bound to 127.0.0.1,
  cross-origin answers narrowed to its own address, its API pages and metrics off, its caches in the plugin's folder.
  LMDeploy takes a key only on its command line, which this plugin never writes one on, so its servers check no key
  (KEYED False) and are never shared beyond this machine from here.
- **Models** (models.py): looked up at the Hub's API (size, files, licence, gate) before anything is downloaded;
  downloaded as a cancellable job with the environment's own hf; listed from the plugin's own models folder with what
  each one's config.json says (its architecture, whether TurboMind serves it, its weights' format); deleted there.

From the command line (reads only): python -m plugin_platform_lmdeploy status | serving | models NAME | launch NAME."""
from __future__ import annotations

from liminal_libs.machines import children
from liminal_libs.machines.ai import HIDDEN, Installed, Package, Refusal
from liminal_libs.machines.endpoint import Endpoint
from liminal_libs.machines.modelserver import Instance, ModelServer, Seen, ServerRun
from liminal_libs.sources.keys import KeyVar

from . import apis, models, release, server
from .apis import APIS
from .release import Here

__version__ = "0.1.1"
__all__ = ["MACHINE", "LmdeployPlugin", "Here", "__version__"]

ABOUT = ("LMDeploy on this PC or another machine: its environment installed, a server per model started and stopped "
         "here by its TurboMind engine alone, on 127.0.0.1.")
TOKEN = KeyVar("LIMINAL_AI_LMDEPLOY_HF_TOKEN", label="Hugging Face token",
               hint=("optional: for gated or private models; it goes to huggingface.co with a lookup and a download "
                     "you ask for, never to a server"))


class LmdeployRun(ServerRun):
    """The plugin's run state (the base's and more): the lookups kept, the last name looked up, the models folder as
    last read, the download under way and its panel, the Hub's last answer, this Linux's C library."""

    def __init__(self):
        super().__init__()
        self.found: list = []  # models.Found, the newest first
        self.last = ""  # the name last looked up (the search box shows it)
        self.cache: tuple | None = None  # (when, [hub.Entry])
        self.pulling = ""  # the repository being downloaded
        # its panel, cancelled when the plugin is turned off or the platform stops: a slot of its own, since the
        # base's (`job`) is an install's, emptied by every install or uninstall
        self.pull_panel = None
        self.hub: dict = {}  # the Hub's last answer, as Endpoint.seen said it
        self.libc: tuple | None = None  # this Linux's C library as Python read it (release.runs_here), once


class LmdeployPlugin(ModelServer):
    """LMDeploy's plugin: its install here, its servers (one model each, TurboMind alone), its models, its instances
    elsewhere."""

    name = "lmdeploy"
    title = "LMDeploy"
    about = ABOUT
    order = 513
    ICON = "lmdeploy"
    api = "2.2"
    apis = APIS
    LIBS_FLOOR = "1.99.0"  # the base's changes page-only (1.99.0), hub.py and SPEAKS (1.66.0)
    DEFAULT_PORT = 23333
    NATIVE = "lmdeploy"
    KEYED = False  # LMDeploy takes keys on its command line alone (--api-keys), and a key never goes on one
    READY_S = 900.0  # a first start loads its model for minutes before it listens
    SPEAKS = ("chat", "completions")  # no Responses API (the user's answer 2: Codex CLI cannot use it)
    MODEL_COLUMNS = ()
    SETTINGS = server.SETTINGS
    HOOKS = (*ModelServer.HOOKS, "find", "clear_found", "pull", "delete_model")
    JOB_HOOKS = (*ModelServer.JOB_HOOKS, "pull")
    CANCEL_HOOKS = (*ModelServer.CANCEL_HOOKS, "pull")
    START_HOOKS = (*ModelServer.START_HOOKS, "pull")  # a download runs the environment's hf: no install meanwhile
    # its own changes asked only from its page (liminal-libs 1.66.0's word): a model downloaded, a downloaded model's
    # files deleted. The base's changes are the library's to say (1.99.0): the names add up along the class line.
    # Its lookup at the Hub (find) and the lookups kept forgotten (clear_found) are the page's too (0.1.1; the user,
    # 2026-11-26, of reads made with a key: "Page-only"): find asks huggingface.co about a name of the asker's choosing
    # with your Hugging Face token where one is kept, and clear_found empties what the Models tab keeps, a choice of
    # the person at the page. So it has no reads of its own: the bases' come by themselves
    PAGE_ONLY_HOOKS = ("pull", "delete_model", "find", "clear_found")
    # what it needs of a machine to serve here (the needs plan's rows, read only once liminal_libs.needs exists): each
    # a part, not a requirement, since an instance elsewhere serves anywhere
    NEEDS = (*getattr(ModelServer, "NEEDS", ()),
             ("architecture", "x86-64", "part", "LMDeploy's wheels are built for x86-64 alone"),
             ("gpu", "nvidia", "part", "TurboMind runs on an NVIDIA GPU"),
             ("gpu.driver", ">=572.61 (Windows), >=570.124.06 (Linux)", "part", "what CUDA 12.8 needs"),
             ("libc", "glibc>=2.28", "part", "its Linux wheels are manylinux_2_28"),
             ("wsl", "none|2", "part", "WSL 1 has no GPU"),
             ("disk", ">=20 GiB", "part", "the install at its peak, uv's cache beside the environment"),
             ("internet", "yes", "part", "the install and a model's download, on a click"))
    INTERNAL_NOTES = (
        "No key is kept for its local instances, since its servers check none (KEYED False: LMDeploy takes a key on "
        "its command line alone). A remote instance's key, where it has one, is a secret pair of its own, "
        "LIMINAL_AI_LMDEPLOY_<INSTANCE>_KEY.",
        "Its rows say it speaks chat and completions: LMDeploy 0.18.0's api_server has no OpenAI Responses API.",
    )
    RUN_STATE = LmdeployRun
    keys = (TOKEN,)

    def __init__(self, outside=None, here: Here | None = None):
        super().__init__(outside)
        self.here = here if here is not None else Here()

    @property
    def KEEPS(self) -> tuple:  # noqa: N802 - a declared word of the contract, as READ_HOOKS
        """What it keeps through the login beside its keys: its instances, and a remote instance's key as its own
        pair."""
        stem = self.variable("instances").rsplit("_", 1)[0]  # LIMINAL_AI_LMDEPLOY
        return ((self.variable("instances"), "its instances: each one's name, where it is, its port, its model and "
                                             "settings, and whether it starts at boot"),
                (f"{stem}_<INSTANCE>_KEY", "a remote instance's key (Key…), a secret pair of its own"))

    # -- what every AI plugin says ------------------------------------------------------------------------------------
    def package(self) -> Package | None:
        return release.package(self)

    def installed(self) -> Installed | None:
        return release.installed(self)

    def install_into(self, log, cancel) -> list[str]:
        return release.install_into(self, log, cancel)

    def remove_from(self) -> list[str]:
        return release.remove_from(self)

    def runs_here(self) -> str:
        return release.runs_here(self)

    def systemd_runs(self) -> bool:
        return bool(self.here.systemd())

    def install(self, accept_licence=False, log=None) -> dict:
        """LMDeploy installed (or updated to the pinned 0.18.0) into this plugin's folder, as a job: the base's,
        refused first while a model is being downloaded (the environment's own hf runs from the folder an install
        replaces)."""
        busy = models.downloading(self)
        if busy:
            raise Refusal(busy)
        return super().install(accept_licence, log)

    def uninstall(self, accept_removal=False) -> dict:
        """LMDeploy removed from this plugin's folder, asked twice: the base's, refused first while a model is being
        downloaded."""
        busy = models.downloading(self)
        if busy:
            raise Refusal(busy)
        return super().uninstall(accept_removal)

    def status(self) -> dict:
        """Its Main (the base's: the install, its instances, main_more), Install… and Uninstall… disabled while a
        model is being downloaded, the download said as why. First, a running server whose log says it fell back to
        the PyTorch engine is stopped (a file read), and said as a problem."""
        try:
            fell = server.stop_fallbacks(self)
        except Exception as e:  # noqa: BLE001 - a status never raises: the Models tab still stops it
            fell = [f"whether a server fell back to PyTorch could not be read: {e}"]
        out = super().status()
        if fell:
            out["problem"] = "; ".join(w for w in (str(out.get("problem") or ""), *fell) if w)
        busy = models.downloading(self)
        if busy:
            for action in out.get("actions") or ():
                if (isinstance(action, dict) and action.get("form") in ("ai-install", "ai-uninstall")
                        and not action.get("disabled")):
                    action.update(disabled=True, note=busy)
        return out

    def app_stopping(self) -> None:
        """The platform stops, or the plugin is turned off: a download under way asked to stop, as the page's Cancel
        asks it (what came stays in the models folder), then the servers a click started stopped (the base's). It
        bears a second call."""
        with self._rt.lock:
            panel = self._rt.pull_panel
        if panel is not None:
            try:
                panel.cancel()
            except Exception:  # noqa: BLE001 - a panel that takes no cancel: the download ends by itself
                pass
        super().app_stopping()

    # -- what a model server says -------------------------------------------------------------------------------------
    def launch(self, instance: Instance, key: str = "") -> children.Launch:
        return server.launch(self, instance, key)

    def look(self, at: Endpoint, instance: Instance) -> Seen:
        return server.look(self, at, instance)

    def start_words(self, instance: Instance) -> str:
        return server.start_words(self, instance)

    def serving(self, name: str = "") -> dict:
        """What it offers the other AI plugins (the base's rows), each local instance's model said as the name its
        server answers to: what a harness sends as its model, never the folder it loads."""
        try:
            server.stop_fallbacks(self)  # never offered to another plugin while it runs on PyTorch
        except Exception:  # noqa: BLE001 - the status says what could not be read
            pass
        out = super().serving(name)
        kept = self.kept_instances().instances
        for row in out.get("instances", []):
            instance = kept.get(row.get("name"))
            if instance is not None and instance.where == "local":
                row["model"] = server.answers_to(instance)
        return out

    def more(self, shown: Instance | None, seen: Seen | None) -> dict:
        return models.page(self, shown, seen)

    def main_more(self) -> dict:
        out: dict = {"tables": [server.leaves(self), release.lacks(self)]}
        if not self.runs_here():
            out["pinned"] = (f"LMDeploy {release.VERSION} (TurboMind; PyTorch {release.TORCH} for CUDA 12.8), "
                             f"uv {release.UV}, {release.PYTHON_SAID}")
        return out

    def api_facts(self) -> dict:
        """What it learned while it ran, by card, asking nothing: what each instance answered, the Hub's last answer
        and the lookups kept, the install here (files looked at), the pins."""
        out: dict = {}
        seen = self.seen_facts()
        if seen:
            out[apis.SERVER] = seen
        with self._rt.lock:
            hub, kept = dict(self._rt.hub), len(self._rt.found)
        if hub.get("at"):
            status = hub.get("status") or "no answer"
            out[apis.HUB] = {"Last asked": f"status {status} in {hub.get('seconds', 0):g} s"
                                           + (f": {hub['error']}" if hub.get("error") else ""),
                             "Lookups kept": str(kept)}
        try:
            have = self.installed()
        except Exception:  # noqa: BLE001 - a fact that cannot be had is left out
            have = None
        if have is not None:
            out[apis.COMMAND] = {"Installed": f"{have.version or 'version not read'} at {have.path}"
                                              + (" (yours, used as it is)" if have.how == "found" else "")}
        out[apis.UV] = {"Pinned": f"uv {release.UV}, {release.PYTHON_SAID}"}
        return out

    # -- the pieces its modules share ---------------------------------------------------------------------------------
    def gpus(self):
        """This machine's NVIDIA cards, the kept reading (nvidia-smi, windowless: a tab and a job, never a status)."""
        return self._gpus()

    def hf_token(self) -> str:
        """The user's Hugging Face token, kept through the login (Settings); "" when none is."""
        return self.pair(TOKEN.name)

    def said(self, words: str, ok: bool = True, lines=()) -> dict:
        """A hook's answer, every secret the plugin knows taken out of it."""
        return self._reply(words, ok, lines)

    def note_hub(self, at: Endpoint) -> None:
        with self._rt.lock:
            self._rt.hub = dict(at.seen)

    def stream_whole(self, argv, line, *, env, cwd=None, timeout: float = 1800.0, cancel=None) -> int:
        """A command run to its end with `env` as its whole environment (uv's, hf's: made here with what they must
        not inherit taken out), windowless, each line to `line`: children.stream, or the Outside's streamer."""
        args = [str(a) for a in argv]
        if self.out.streamer is not None:
            return self.out.streamer(args, line, env=dict(env), cwd=cwd, timeout=timeout, cancel=cancel)
        return children.stream(args, line, env=dict(env), cwd=cwd, timeout=timeout, cancel=cancel,
                               popen=self.out.popen, system=self.system)

    def run_whole(self, argv, log, *, env, timeout: float, cancel=None, cwd=None, secret=()) -> int:
        """`stream_whole` with its lines to a job's panel, every secret named taken out of them first."""
        hidden = sorted({s for s in secret if isinstance(s, str) and len(s) >= 4}, key=len, reverse=True)

        def line(text: str) -> None:
            for value in hidden:
                text = text.replace(value, HIDDEN)
            if callable(log):
                try:
                    log(text)
                except Exception:  # noqa: BLE001 - the work goes on without its panel
                    pass

        stop = cancel if cancel is not None else self.cancel_of(log)
        return self.stream_whole(argv, line, env=env, cwd=cwd, timeout=timeout, cancel=stop)

    # -- its own hooks ------------------------------------------------------------------------------------------------
    def find(self, model) -> dict:
        """A model looked up at the Hub's API before anything is downloaded: its files, its size and its licence."""
        return models.find(self, model)

    def clear_found(self) -> dict:
        """The lookups kept on the Models tab forgotten."""
        return models.clear_found(self)

    def pull(self, model, accept_licence=False, log=None) -> dict:
        """A model downloaded into this plugin's models folder by the environment's hf, as a job its user may
        cancel."""
        return models.pull(self, model, accept_licence, log)

    def delete_model(self, model) -> dict:
        """A downloaded model's files deleted from this plugin's models folder, unless a running server serves it."""
        return models.delete_model(self, model)


MACHINE = LmdeployPlugin()
