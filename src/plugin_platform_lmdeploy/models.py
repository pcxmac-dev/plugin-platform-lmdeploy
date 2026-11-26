"""LMDeploy's models: looked up, downloaded, listed and deleted here (plugin-platform-lmdeploy 0.1.0), on the library's
Hugging Face pieces (liminal_libs.machines.hub, 1.66.0: the cache's layout, a download's progress, a delete inside
the plugin's folder, a card's licence and gate, hf's environment).

LMDeploy has no download, delete, load or unload over its API: one server serves one model, read from a folder when
it starts. So this plugin keeps a models folder of its own, home/models (HF_HOME; the cache is its hub/, in Hugging
Face's layout), and:

- **Looks a model up** at the Hub's API before anything is downloaded (`find`): its files, its exact size
  (GET /api/models/{repo}/treesize/main) and its licence and gating (GET /api/models/{repo}, the model's card data),
  and whether TurboMind serves its architecture (its config.json is not read before a download: the Hub's card data
  says the architecture only sometimes, so that is said after it). It needs nothing installed. Each lookup is kept
  an hour, 12 at most, newest first, for the table and for the download's second confirmation, which is built from
  it without asking again.
- **Downloads** it as a job (`pull`) with the environment's own `hf download <repo> --quiet` into the models folder
  (hub.child_env: hf told its folder, its telemetry, its update check, its progress bars and the Hub's shared store
  off, the user's token only where one is kept, as HF_TOKEN, never on a command line), its progress said from the
  folder's bytes every 2 s (hub.Watch), Cancel ending hf (what came stays in the folder, as hf leaves it), and so does
  turning the plugin off.
- **Lists** what is downloaded (`cached`), read from the folder itself every 60 s at most (hub.read), with what each
  one's config.json says: its architecture, whether TurboMind serves it, and its weights' format (AWQ, GPTQ,
  compressed-tensors, FP8 or MXFP4, from its quantization_config).
- **Deletes** one (`delete_model`, hub.delete), refused while a running server of this plugin serves it.

Each system keeps its own models on its own disk: with the folder on another system's drive (mounts.foreign), a
download and a delete are refused in words, as Install and Start are."""
from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from liminal_libs import mounts
from liminal_libs.machines import hub as hf
from liminal_libs.machines.ai import ACCEPT_LICENCE, Refusal, exactly
from liminal_libs.machines.endpoint import Endpoint, EndpointError
from liminal_libs.machines.plugin import CommandError
from liminal_libs.machines.terminal import Cancelled
from liminal_libs.machines.world import size_words

from . import release, server

WHERE = "huggingface.co"
FOUND_S = 3600.0  # a lookup is kept an hour
FOUND_MOST = 12  # lookups kept at once, the newest first
FIND_S = 15.0  # an ask of the Hub's API waits at most this long
CACHE_S = 60.0  # the models folder is read again at most this often
PULL_S = 21600.0  # a download stops by itself after 6 hours
ROOM = 1 << 30  # what a download leaves free besides its own size
PLACEHOLDER = "Qwen/Qwen3-0.6B, or any org/name of Hugging Face's"
# taken out of the platform's environment for hf besides the library's own (hub.DROP) and Liminal's
HF_DROP = ("LMDEPLOY_*", "MODELSCOPE_*", "TORCH_*", "TRITON_*", "RAY_*")
# a local instance's settings that are its model's own: Serve it… leaves them empty, so they follow a new model
FOLLOW = ("served", "context", "format")
FORMATS_SAID = {"awq": "AWQ", "gptq": "GPTQ", "compressed-tensors": "compressed-tensors", "fp8": "FP8",
                "mxfp4": "MXFP4"}


def hub(p) -> Path:
    """The plugin's Hugging Face cache: home/models/hub."""
    return p.home / "models" / hf.FOLDER


def page_of(repo: str) -> str:
    return f"{release.HUB}/{repo}"


def _away(p) -> str:
    """"" or why this system does not fill or empty the models folder: the plugin's folder (home_problem), or its
    models folder through a link, on another system's drive (mounts.foreign). A mount table read, nothing run."""
    trouble = p.home_problem()
    if trouble:
        return trouble
    try:
        mounts.owner_only(hub(p), "LMDeploy's models are downloaded and deleted by the system whose disk holds them")
    except mounts.NotHere as e:
        return " ".join(str(e).split())
    except Exception:  # noqa: BLE001 - a path that cannot be judged is this system's (home_problem's rule)
        return ""
    return ""


def downloading(p) -> str:
    """"" or, while a model is being downloaded, that in words: what Install… and Uninstall… wait for, since the
    environment's own hf runs from the folder they would change."""
    with p._rt.lock:
        busy = p._rt.pulling
    return f"{busy} is being downloaded by LMDeploy's hf: Cancel it in its panel first, or let it end" if busy else ""


@dataclass(frozen=True)
class Found:
    """A model looked up: its repository as the Hub names it, where it was read, its page, when, its size in bytes
    (None: not said), its files, its licence in words, whether it is gated, why what is missing is missing, and what
    fits this machine's GPU as it was read then (the download's second confirmation says it without reading again)."""

    model: str
    where: str
    page: str
    at: float
    size: int | None = None
    files: int = 0
    licence: str = ""
    gated: bool = False
    why: str = ""
    fit: str = ""


# -- the models folder -------------------------------------------------------------------------------------------------
def entries(p, fresh: bool = False) -> list[hf.Entry]:
    """The models folder as last read, read again once CACHE_S old (`fresh`: now)."""
    now = p.out.clock()
    with p._rt.lock:
        kept = p._rt.cache
    if kept is not None and not fresh and 0 <= now - kept[0] < CACHE_S:
        return list(kept[1])
    found = hf.read(hub(p))
    with p._rt.lock:
        p._rt.cache = (now, found)
    return list(found)


def forget_cache(p) -> None:
    with p._rt.lock:
        p._rt.cache = None


def cached(p, repo: str, fresh: bool = False) -> hf.Entry | None:
    """The repository downloaded here by that name (its case as typed or as its folder spells it); None for none."""
    return hf.find(entries(p, fresh), repo)


def snapshot_of(entry: hf.Entry) -> Path | None:
    return entry.path / "snapshots" / entry.snapshot if entry.snapshot else None


def format_hint(config: Mapping) -> str:
    """The weights' format a model's config.json says (AWQ, GPTQ, compressed-tensors, FP8, MXFP4), for Change…'s
    Weights' format; "" for plain weights or none said."""
    q = server.quantization(config)
    return FORMATS_SAID.get(q, q.upper() if q else "")


def what_it_is(entry: hf.Entry) -> tuple[str, str]:
    """A downloaded model's architecture with whether TurboMind serves it, and its format hint: from its snapshot's
    config.json alone (a file read)."""
    folder = snapshot_of(entry)
    if folder is None:
        return "not read: no snapshot", ""
    why = server.turbomind_refusal(folder, entry.repo)
    config = server.config_of(folder)
    arch = server.architecture(config) or "no architecture named"
    return (f"{arch}: " + ("TurboMind serves it" if not why else "TurboMind does not serve it: it is not started "
                                                                  "here")), format_hint(config)


# -- the GPU -----------------------------------------------------------------------------------------------------------
def fit_words(size, gpus, share: int, name: str = "a server") -> str:
    """Whether a model's weights fit the GPU's memory, and what its cache then takes, in words: TurboMind loads the
    weights, then keeps `share`% of what is free for its cache until it stops."""
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        return "its size is not said"
    if gpus is None or not gpus.cards:
        why = getattr(gpus, "why", "") or "nvidia-smi says nothing"
        return f"no NVIDIA GPU is read here ({why}): what fits is not known"
    card = max(gpus.cards, key=lambda g: (g.total, g.free))
    if size >= card.free:
        return (f"the weights ({size_words(size)}) are larger than what the {card.name} has free now "
                f"({size_words(card.free)}): LMDeploy would not start with it (a smaller or a quantized model)")
    cache = (card.free - size) * share // 100
    return (f"the weights ({size_words(size)}) fit in the {card.name}'s {size_words(card.free)} free now; {name} then "
            f"keeps {share}% of what is left ({size_words(cache)}) for its cache until it stops")


def _share_shown(p, shown) -> tuple[int, str]:
    if shown is not None and shown.where == "local":
        return server.share_of(shown), shown.name
    return server.SHARE, "a server"


# -- looking a model up ------------------------------------------------------------------------------------------------
def _keep(p, found: Found) -> None:
    with p._rt.lock:
        kept = [f for f in p._rt.found if f.model.lower() != found.model.lower()]
        p._rt.found = [found, *kept][:FOUND_MOST]
        p._rt.last = found.model


def lookups(p) -> list[Found]:
    """The lookups kept, the newest first: an hour at most."""
    now = p.out.clock()
    with p._rt.lock:
        p._rt.found = [f for f in p._rt.found if 0 <= now - f.at < FOUND_S]
        return list(p._rt.found)


def lookup(p, repo: str) -> Found | None:
    wanted = str(repo or "").strip().lower()
    return next((f for f in lookups(p) if f.model.lower() == wanted), None)


def find(p, model) -> dict:
    """A model looked up at the Hub's API: its files, its exact size and its licence, kept for its download. A read:
    nothing is downloaded or kept beyond this plugin's run state."""
    repo = hf.repository(model)
    token = p.hf_token()
    at = Endpoint(release.HUB, key=token, timeout=FIND_S, transport=p.out.transport)
    now = p.out.clock()
    try:
        info = at.get(f"/api/models/{repo}", params={"expand": list(hf.EXPAND)})
        if not isinstance(info, Mapping):
            raise EndpointError(f"GET /api/models/{repo}: its answer is not a model's information")
    except EndpointError as e:
        why = hf.why(e, token, WHERE)
        _keep(p, Found(repo, WHERE, page_of(repo), now, why=why))
        p.note_hub(at)
        return p.said(f"{repo}: {WHERE} did not say its size or licence ({why})", False,
                      [f"asked at {WHERE}: a download of it would be made unseen"])
    named = info.get("id") if isinstance(info.get("id"), str) else repo
    canonical = named if hf.REPO.fullmatch(named or "") else repo
    size, missing = None, ""
    try:
        size = hf.size_of(at.get(f"/api/models/{canonical}/treesize/main"))
        missing = "" if size else "its size was not in the Hub's answer"
    except EndpointError as e:
        missing = f"its size was not said: {hf.why(e, token, WHERE)}"
    p.note_hub(at)
    files = hf.files_of(info)
    licence, where = hf.licence_of(info, page_of(canonical))
    gated = hf.gated(info.get("gated"))
    share, name = _share_shown(p, p.shown())
    fit = fit_words(size, p.gpus(), share, name)
    _keep(p, Found(canonical, WHERE, where, now, size, files, licence, gated, missing, fit))
    lines = [fit, f"read at {WHERE}",
             "whether TurboMind serves it is read from its config.json once it is downloaded (Downloaded here)"]
    if gated:
        lines.append("gated: its files need the Hugging Face token of an account given access"
                     + ("; the one kept here is sent with its download" if token else " (Settings): none is kept"))
    if missing:
        lines.append(missing)
    return p.said(f"{canonical}: {server.size_said(size)} to download in {files} files; licence: "
                  f"{licence or 'not said'}", True, lines)


def clear_found(p) -> dict:
    with p._rt.lock:
        p._rt.found, p._rt.last = [], ""
    return p.said("the lookups are forgotten")


# -- downloading -------------------------------------------------------------------------------------------------------
def hf_environment(p, token: str) -> dict[str, str]:
    """hf's whole environment (hub.child_env): the platform's without the Hub's, Transformers', Python's, LMDeploy's,
    PyTorch's and Ray's own variables and every pair of Liminal's; its folder the plugin's, its telemetry, its update
    check, its progress bars and the shared store off, the user's token only where one is kept."""
    return hf.child_env(p.home / "models", token, base=p.out.env, drop=(*p.dropped(), *HF_DROP), system=p.system)


def _second_ask(found: Found, folder: Path, free: int) -> str:
    """The words of a download's second confirmation, from its kept lookup alone (nothing asked, nothing run: it is
    said within the platform's one second)."""
    if found.why and not found.size and not found.licence:
        return (f"{found.where} did not say {found.model}'s size or licence ({found.why}): it would be downloaded "
                "unseen.")
    licence = found.licence or "not said"
    gated = " It is gated: your token is sent with the download." if found.gated else ""
    fit = f" {found.fit[0].upper()}{found.fit[1:]}." if found.fit else ""
    return (f"{found.model}: {server.size_said(found.size)} in {found.files} files from {found.where}, into {folder} "
            f"({size_words(free)} free). Licence: {licence}; its words are at {found.page}.{fit}{gated} Whether "
            "TurboMind serves it is read from its config.json once it is here.")


class _Panel:
    """The lines and the cancel of a download asked without a job's panel (a test, another plugin): a cancel of its
    own, which a detach sets."""

    def __init__(self, log):
        self._log, self._stop = log, threading.Event()

    def __call__(self, line: str) -> None:
        if callable(self._log):
            try:
                self._log(line)
            except Exception:  # noqa: BLE001 - the download goes on without its panel
                pass

    def cancel(self) -> None:
        self._stop.set()

    @property
    def cancelled(self) -> bool:
        return self._stop.is_set() or bool(getattr(self._log, "cancelled", False))


def pull(p, model, accept_licence=False, log=None) -> dict:
    """A model downloaded into this plugin's models folder as a job: refused before its first line where the name is
    none, LMDeploy does not run here, the folder is on another system's drive, LMDeploy is not installed here, it was
    not looked up, it is gated and no token is kept, or the folder has not the room; then asked a second time with its
    size and licence from the lookup (accept_licence). Its panel is kept in a slot of its own (`pull_panel`), so
    turning the plugin off cancels it whatever else was asked meanwhile."""
    repo = hf.repository(model)
    here = p.runs_here()
    if here:
        raise Refusal(here)
    away = _away(p)
    if away:
        raise Refusal(away)
    have = p.installed()
    if have is None:
        raise Refusal("install LMDeploy first (Install…): its environment's hf command downloads its models")
    program = release.hf_of(p, have)
    if not program.is_file():
        raise Refusal(f"there is no hf in the environment of {have.path}: Hugging Face's command comes with "
                      "LMDeploy's environment")
    found = lookup(p, repo)
    if found is None:
        raise Refusal(f"look {repo} up first (Find a model): its size and licence are read before a download")
    token = p.hf_token()
    if found.gated and not token:
        raise Refusal(f"{found.model} is gated: it needs the Hugging Face token of an account that was given access "
                      "(Settings)")
    folder = hub(p)
    free = p.here.free_at(folder)
    if found.size and free < found.size + ROOM:
        raise Refusal(f"{found.model} needs {size_words(found.size)}; {folder} has {size_words(free)} free")
    with p._rt.lock:
        busy = p._rt.pulling
    if busy:
        raise Refusal(f"{busy} is being downloaded: one download at a time")
    if not exactly(accept_licence):
        raise Refusal(_second_ask(found, folder, free), accept=ACCEPT_LICENCE)
    panel = log if callable(getattr(log, "cancel", None)) else _Panel(log)
    with p._rt.lock:
        if p._rt.pulling:
            raise Refusal(f"{p._rt.pulling} is being downloaded: one download at a time")
        p._rt.pulling, p._rt.pull_panel = found.model, panel
    try:
        return _download(p, found, program, token, panel)
    finally:
        with p._rt.lock:
            p._rt.pulling = ""
            if p._rt.pull_panel is panel:
                p._rt.pull_panel = None
        forget_cache(p)


def _download(p, found: Found, program: Path, token: str, panel) -> dict:
    target, folder = hf.folder_of(hub(p), found.model), hub(p)
    panel(f"{found.model} is downloaded from {WHERE} (and the storage it sends each file on to) by LMDeploy's hf "
          f"command into {folder}; nothing is started")
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return p.said(f"{found.model} was not downloaded: {folder} could not be made ({' '.join(str(e).split())})",
                      False)
    watch = hf.Watch(target, found.size, found.model, panel)
    try:
        with watch:
            code = p.run_whole(hf.download_argv(program, found.model), panel, env=hf_environment(p, token),
                               timeout=PULL_S, cancel=p.cancel_of(panel), cwd=str(folder), secret=(token,))
    except Cancelled:
        return p.said(f"cancelled: what came of {found.model} stays in {target} (hf's .incomplete files), and "
                      "Delete… removes it", False)
    except (CommandError, OSError, ValueError) as e:
        return p.said(f"{found.model} was not downloaded: {' '.join(str(e).split())}", False)
    if code != 0:
        return p.said(f"{found.model} was not downloaded: hf ended with {code} (its last lines are in the panel above)",
                      False)
    entry = cached(p, found.model, fresh=True)
    if entry is None or entry.incomplete or not entry.snapshot:
        return p.said(f"hf ended, but {found.model} is not whole in {hub(p)}", False)
    watch.end(f"{found.model}: {size_words(entry.size)} in {entry.files} files")
    kind, form = what_it_is(entry)
    lines = [f"in {entry.path}", kind]
    if form:
        lines.append(f"its weights are {form}: Change… may name that format (LMDeploy reads it from the files too)")
    lines.append(f"serve it: Serve it… on the Models tab, or an instance's Model ({entry.repo})")
    return p.said(f"{entry.repo} is here ({size_words(entry.size)})", True, lines)


# -- deleting ----------------------------------------------------------------------------------------------------------
def _serving(p, repo: str) -> tuple[list[str], list[str]]:
    """The local instances set to serve `repo`: (those running now, the rest)."""
    wanted = repo.lower()
    users = [i for i in p.kept_instances().instances.values()
             if i.where == "local" and str(i.settings.get("model") or "").strip().lower() == wanted]
    running = set(p.running())
    return [i.name for i in users if i.name in running], [i.name for i in users if i.name not in running]


def delete_model(p, model) -> dict:
    """A model's files deleted from this plugin's models folder (its repository's folder, links as links, and its
    lock files). Refused: the folder on another system's drive; not downloaded here; being downloaded; served by a
    running server of this plugin."""
    repo = hf.repository(model)
    away = _away(p)
    if away:
        raise Refusal(away)
    entry = cached(p, repo, fresh=True)
    if entry is None:
        raise Refusal(f"{repo} is not downloaded here: there is nothing to delete")
    with p._rt.lock:
        busy = p._rt.pulling
    if busy and busy.lower() == entry.repo.lower():
        raise Refusal(f"{entry.repo} is being downloaded: Cancel it in the panel first")
    running, idle = _serving(p, entry.repo)
    if running:
        raise Refusal(f"stop {', '.join(running)} first: it serves {entry.repo}")
    try:
        freed, lines = hf.delete(entry.path, p.home / "models")
    except (OSError, ValueError, mounts.NotHere) as e:
        forget_cache(p)
        return p.said(f"{entry.repo} is not deleted whole: {' '.join(str(e).split())}", False)
    if idle:
        lines.append(f"{', '.join(idle)} {'is' if len(idle) == 1 else 'are'} set to serve it: download it again before "
                     f"{'it starts' if len(idle) == 1 else 'they start'}")
    forget_cache(p)
    return p.said(f"{entry.repo} is deleted from {hub(p)} ({size_words(freed)} freed)", True, lines)


# -- the Models tab's part ---------------------------------------------------------------------------------------------
def _find_table(p, shown, can: str) -> dict:
    gpus = p.gpus()
    share, name = _share_shown(p, shown)
    with p._rt.lock:
        last = p._rt.last
    rows = []
    token = p.hf_token()
    for f in lookups(p):
        why = can or (f"{f.model} is gated, and no Hugging Face token is kept (Settings)" if f.gated and not token
                      else "")
        cell = {"form": "model-pull", "title": "Download…", "values": {"model": f.model},
                "label": f"Download {f.model}"}
        if why:
            cell.update(disabled=True, note=why)
        sub = f.where + (" · gated" if f.gated else "") + (f" · {f.why}" if f.why else "")
        rows.append([{"text": f.model, "sub": sub}, server.size_said(f.size), f.licence or "not said",
                     fit_words(f.size, gpus, share, name) if f.size else "–", cell])
    kept = "kept: it is sent with a lookup and a download" if token else "none kept (Settings)"
    return {"title": "Find a model",
            "search": {"form": "model-find", "field": "model", "value": last, "placeholder": PLACEHOLDER},
            "note": (f"Looked up at {WHERE}'s API: a model's files, its size and its licence, before anything is "
                     f"downloaded. Your Hugging Face token: {kept}."),
            "columns": [{"title": "Model"}, {"title": "Download", "num": True}, {"title": "Licence"},
                        {"title": "Fits this GPU"}, {"title": ""}],
            "rows": rows,
            "empty": (f"Type a model's name above (org/name), then Look up: its size and licence are read at {WHERE} "
                      "before anything is downloaded."),
            "actions": [{"hook": "clear_found", "args": [], "title": "Clear", "note": "forgets these lookups"}]}


def _serve_cell(p, shown, e: hf.Entry, running) -> tuple[dict, str]:
    """Serve it… of a repository downloaded here, and why it cannot be asked now ("" when it can). For a local
    instance shown: Change… of it, opened with its own values as the base's Change… opens it, its model this one and
    the settings that are its old model's own (FOLLOW) left empty. Otherwise Add an instance… that serves it."""
    if shown is None or shown.where != "local":
        return ({"form": "ai-add", "title": "Serve it…", "values": {"where": "local", "model": e.repo},
                 "label": f"Add an instance that serves {e.repo}"}, "")
    values = {**p._change_values(shown), "model": e.repo, **dict.fromkeys(FOLLOW)}
    cell = {"form": "ai-change", "title": "Serve it…", "values": values, "label": f"Serve {e.repo} on {shown.name}",
            "note": (f"Change… of {shown.name} with {e.repo} as its model, on its own port {p.port_of(shown)}: its "
                     "cache share and its tool call parser stay; the name it answers to, its session length and its "
                     "weights' format are left empty, to follow the new model")}
    if str(shown.settings.get("model") or "").strip().lower() == e.repo.lower():
        return cell, f"{shown.name} is set to serve it already (Change… on Main changes its settings)"
    if shown.name in running:
        return cell, f"stop {shown.name} first: its model is read when it starts"
    return cell, ""


def _cache_table(p, shown, can: str) -> dict:
    """Downloaded here; `can`: "" or why nothing of it can be asked now (the folder on another system's drive)."""
    gpus = p.gpus()
    share, name = _share_shown(p, shown)
    listed = p.kept_instances().instances.values()
    running = set(p.running())
    rows = []
    for e in entries(p):
        users = [i.name + (" (running)" if i.name in running else "") for i in listed
                 if i.where == "local" and str(i.settings.get("model") or "").strip().lower() == e.repo.lower()]
        kind, form = what_it_is(e)
        sub = ("incomplete: a download was stopped" if e.incomplete else e.why or kind)
        serve, why = _serve_cell(p, shown, e, running)
        why = can or why
        if e.incomplete:
            why = why or "its download is not whole: download it again first"
        folder = snapshot_of(e)
        refused = server.turbomind_refusal(folder, e.repo) if folder is not None and not e.incomplete else ""
        why = why or refused
        if why:
            serve.update(disabled=True, note=why)
        delete = {"form": "model-delete", "title": "Delete…", "values": {"model": e.repo},
                  "label": f"Delete {e.repo}"}
        busy = [u for u in users if u.endswith("(running)")]
        if can:
            delete.update(disabled=True, note=can)
        elif busy:
            delete.update(disabled=True, note=f"stop {', '.join(u.split(' ')[0] for u in busy)} first: it serves it")
        size = size_words(e.size) + ("" if e.whole else "+")
        rows.append([{"text": e.repo, "sub": sub}, size, form or "plain", fit_words(e.size, gpus, share, name),
                     ", ".join(users) or "–", serve, delete])
    return {"title": "Downloaded here",
            "note": (f"This plugin's own models folder, {hub(p)} (Hugging Face's layout), read from the folder itself, "
                     "every minute at most. A server here is started offline with one of them, by TurboMind alone."),
            "columns": [{"title": "Repository"}, {"title": "Size", "num": True}, {"title": "Format"},
                        {"title": "Fits this GPU"}, {"title": "Served by"}, {"title": ""}, {"title": ""}],
            "rows": rows,
            "empty": "Nothing is downloaded here yet: look a model up above, then Download…."}


ELSEWHERE = ("An LMDeploy elsewhere is not started, stopped, set up or given a model from here. Its engine, its "
             "address and its cross-origin answers are its own settings there, and cannot be told from here.")


def _elsewhere_table(shown, seen) -> dict:
    served = ", ".join(str(m["id"]) for m in seen.models) if seen is not None and seen.ok and seen.models else ""
    return {"title": f"{shown.name} runs elsewhere", "note": ELSEWHERE,
            "columns": [{"title": "What"}, {"title": "Here"}],
            "rows": [["Started, stopped, set up", "where it runs: not from here"],
                     ["Its model", served or "the one it was started with, as it lists it above"],
                     ["Its engine", "its own setting where it was started: not told from here"]]}


def forms() -> list[dict]:
    """The Models tab's own forms: a lookup (a read), a download (a job, its licence accepted a second time) and a
    delete (a danger)."""
    return [
        {"id": "model-find", "hook": "find", "title": "Look a model up",
         "fields": [{"name": "model", "label": "Model", "type": "text", "required": True, "placeholder": PLACEHOLDER,
                     "note": "org/name, as Hugging Face names it"}],
         "confirm": False},
        {"id": "model-pull", "hook": "pull", "title": "Download a model", "job": True,
         "fields": [{"name": "model", "label": "Model", "type": "fixed"},
                    {"name": ACCEPT_LICENCE, "type": "accept", "title": "Download it under its licence?",
                     "yes": "Accept and download",
                     "words": "{model} is downloaded under the licence named above. Cancel in the panel stops it."}],
         "confirm": {"title": "Download {model}?", "yes": "Go on",
                     "words": ("{model} is downloaded from huggingface.co (and the storage the Hub sends each file on "
                               "to) by LMDeploy's own hf command into this plugin's models folder; nothing is "
                               "started. Its size and licence are said next, before anything is downloaded. Its "
                               "progress shows in the panel below; Cancel stops it, and it stops by itself after 6 "
                               "hours.")}},
        {"id": "model-delete", "hook": "delete_model", "title": "Delete a model",
         "fields": [{"name": "model", "label": "Model", "type": "fixed"}],
         "confirm": {"title": "Delete {model}?", "yes": "Delete", "danger": True,
                     "words": ("Its files are deleted from this plugin's models folder: this cannot be undone, and "
                               "only a download brings them back. An instance set to serve it starts again only "
                               "once it is downloaded again.")}},
    ]


def page(p, shown, seen) -> dict:
    """What LMDeploy adds to the Models tab (more): Find a model and Downloaded here; for an instance that runs
    elsewhere, what is not done from here. Files and kept readings, the GPU's kept reading."""
    tables: list[dict] = []
    if not p.runs_here():
        away = _away(p)
        have = p.installed()
        can = away or ("" if have is not None else "install LMDeploy first (Install…): its own hf command downloads")
        tables += [_find_table(p, shown, can), _cache_table(p, shown, away)]
    if shown is not None and shown.where != "local":
        tables.append(_elsewhere_table(shown, seen))
    return {"tables": tables, "forms": forms() if tables and not p.runs_here() else [], "actions": []}


# the calls this module makes, by card (apis.py's names)
CALLS = {
    "Hugging Face Hub API": ("GET /api/models/{repo}", "GET /api/models/{repo}/treesize/main"),
    "hf, Hugging Face's command": ("hf download",),
}
