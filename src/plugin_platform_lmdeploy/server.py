"""An LMDeploy server: its settings, how it is started, how it is recognised, and what leaves this machine
(plugin-platform-lmdeploy 0.1.0).

One server serves one model, named when it starts, so a local instance's settings are its model, the name it answers
to, its weights' format, its session length, its share of the GPU's free memory for its cache and its tool call
parser. It is started

    <lmdeploy> serve api_server <the model's folder> --server-name 127.0.0.1 --server-port <port>
               --model-name <name> --backend turbomind --tp 1 --cache-max-entry-count 0.70
               [--session-len N] [--model-format F] [--tool-call-parser P]
               --allow-origins http://127.0.0.1:<port> --allowed-media-domains localhost
               --disable-fastapi-docs --disable-metrics --log-level WARNING --max-log-len 0
               --distributed-executor-backend uni

What each switch is for (the pre-build checks' C-L1, read in LMDeploy 0.18.0's sources):

- `--server-name 127.0.0.1` always: LMDeploy's own default is every interface (0.0.0.0). LMDeploy takes a key only on
  its command line (--api-keys), where every program of this machine could read it, so this plugin gives it none
  (KEYED is False) and never shares a server beyond this machine.
- `--allow-origins` its own address: LMDeploy's own default lets every web page read it ('*'), and its command line
  cannot turn cross-origin answers off, only narrow them to one origin. Its docs and its metrics are switched off.
- `--backend turbomind`, and a model TurboMind does not serve refused before the start: LMDeploy's command falls back
  to its PyTorch engine with a warning alone, and that engine always starts a second process holding an
  unauthenticated loopback port (C-L1 4b). Its log is read for that fallback all the same, and a server that fell
  back is stopped (`fell_back`). `--tp 1`: TurboMind on one GPU is one process with one port, and nothing else.
- `--allowed-media-domains localhost`: a vision model fetches the images a request names from any public host
  unless told otherwise; this list holds no host such a fetch may reach.
- `--log-level WARNING` and `--max-log-len 0`: no prompt and no answer reaches its log (they are logged only at its
  REQUEST level, below INFO). UVICORN_LOG_LEVEL=warning drops its per-request lines.
- `--distributed-executor-backend uni` (and LMDEPLOY_EXECUTOR_BACKEND=uni): the PyTorch engine's executor without Ray,
  should LMDeploy ever start it all the same.

It is started offline (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1: a start never downloads; its model is downloaded on
the Models tab first, and the snapshot's whole path is what it is given), away from ModelScope and openMind's hub,
with GPT-OSS's tokenizer vocabulary read from the plugin's own folder alone (TIKTOKEN_ENCODINGS_BASE), its caches in
the plugin's folder, and what the platform's environment holds of LMDeploy's, the Hub's, Transformers', PyTorch's,
Triton's, Ray's and Python's own variables taken out first, with every pair of Liminal's."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

from liminal_libs import mounts
from liminal_libs.machines import children
from liminal_libs.machines.ai import Refusal, Setting
from liminal_libs.machines.endpoint import EndpointError
from liminal_libs.machines.modelserver import Seen
from liminal_libs.machines.world import size_words

from . import release

SHARE = 70  # the share of the GPU's free memory its cache takes, in percent (LMDeploy's own is 80)
SHARE_RANGE = (10, 95)
CONTEXT_RANGE = (1024, 1_048_576)
LOOPBACK = "127.0.0.1"
MEDIA = "localhost"  # --allowed-media-domains: no public host is ever named so
# taken out of the platform's environment for a server (every pair of Liminal's besides, by the base)
DROP = ("LMDEPLOY_*", "HF_*", "HUGGINGFACE_*", "HUGGING_FACE_HUB_TOKEN", "TRANSFORMERS_*", "MODELSCOPE_*", "TORCH_*",
        "TORCHINDUCTOR_*", "TRITON_*", "RAY_*", "TIKTOKEN_*", "UVICORN_*", "TM_*", "SSL_KEYFILE", "SSL_CERTFILE",
        "PYTHON*", "XDG_CACHE_HOME")
REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")  # a Hub repository: org/name
SERVED = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")  # the name it answers to: no slash, no space
# the formats `--model-format` takes (its command line's page)
FORMATS = ("hf", "awq", "gptq", "compressed-tensors", "fp8", "mxfp4")
# the tool call parsers LMDeploy 0.18.0 registers (C-L1 1h): checked by its server at its start
PARSERS = (("qwen3coder", "qwen3coder", "Qwen3-Coder and Qwen3.5"), ("qwen3", "qwen3", "Qwen3"),
           ("qwen2d5", "qwen2d5", "Qwen2.5"), ("qwen", "qwen", "Qwen"), ("llama3", "llama3", "Llama 3"),
           ("internlm", "internlm", "InternLM 2 and 3"), ("intern-s1", "intern-s1", "Intern-S1"),
           ("interns2-preview", "interns2-preview", "Intern-S2-Preview"), ("glm47", "glm47", "GLM-4.7"),
           ("deepseek-v32", "deepseek-v32", "DeepSeek-V3.2"), ("deepseek-v4", "deepseek-v4", "DeepSeek-V4"),
           ("kimi-k2", "kimi-k2", "Kimi K2"))
# the architectures TurboMind serves at 0.18.0 (lmdeploy/turbomind/supported_models.py; C-L1 1g): any other model,
# and one quantized by SmoothQuant, LMDeploy's command would hand to its PyTorch engine
TURBOMIND = ("Qwen2ForCausalLM", "Qwen2MoeForCausalLM", "Qwen2VLForConditionalGeneration",
             "Qwen2_5_VLForConditionalGeneration", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM",
             "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration",
             "InternS2PreviewForConditionalGeneration", "InternS2MobiusForConditionalGeneration", "InternVLChatModel",
             "InternVLForConditionalGeneration", "InternS1ForConditionalGeneration", "LlamaForCausalLM",
             "InternLM2ForCausalLM", "InternLM3ForCausalLM", "Glm4MoeLiteForCausalLM", "GptOssForCausalLM",
             "MixtralForCausalLM")
GPT_OSS = "GptOssForCausalLM"
TIKTOKEN = "o200k_base.tiktoken"  # GPT-OSS's vocabulary, which openai-harmony downloads on its first request
TIKTOKEN_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"  # what openai-harmony 0.0.8 checks
FALLBACK = "Fallback to pytorch engine"  # LMDeploy's warning when its command hands a model to PyTorch
LOG_TAIL = 256 << 10  # the end of a log read for that warning
OWNER = "lmdeploy"  # what its /v1/models says of each model (owned_by): how LMDeploy is told apart
NOT_YET = "it runs but does not answer yet: LMDeploy loads its model before it serves. Log shows where it is."


def repository(text) -> str:
    """A Hub repository's name as typed, checked: `org/name`, each part letters, digits, . _ -, never .. or --
    (the Hub's own rule). Refusal in words otherwise."""
    said = " ".join(str(text or "").split())
    if not REPO.fullmatch(said) or ".." in said or "--" in said:
        raise Refusal(f"a model on Hugging Face is named org/name (Qwen/Qwen3-0.6B): letters, digits, '.', '_' and "
                      f"'-', not {said[:80]!r}")
    return said


def _absolute(text: str) -> bool:
    return os.path.isabs(text) or text.startswith("/") or bool(re.match(r"^[A-Za-z]:[\\/]", text))


class ModelSetting(Setting):
    """The model a server serves: a repository downloaded here (org/name) or a folder's whole path."""

    def clean(self, value):
        text = super().clean(value)
        if text is None:
            return None
        if _absolute(text):
            return text
        return repository(text)


class ServedSetting(Setting):
    """The name the server answers to: no slash and no space (a harness names its model so)."""

    def clean(self, value):
        text = super().clean(value)
        if text is not None and not SERVED.fullmatch(text):
            raise Refusal(f"{self.label} is letters, digits, '.', '_', ':' and '-', with no slash or space: not "
                          f"{text[:60]!r}")
        return text


SETTINGS = (
    ModelSetting("model", "Model", "text",
                 note="a repository downloaded here (org/name, see Models), or a folder's whole path on this system"),
    ServedSetting("served", "Name it answers to", "text",
                  note="empty: the repository's last part in lower case; never with a slash"),
    Setting("format", "Weights' format", "choice", choices=FORMATS,
            note="empty: LMDeploy reads it from the model's own files; the Models tab says what a download holds"),
    Setting("context", "Session length (tokens)", "number", min=CONTEXT_RANGE[0], max=CONTEXT_RANGE[1],
            note="empty: the model's own; a coding harness wants 32,768 or more"),
    Setting("share", "Cache share (percent of the free GPU memory)", "number", default=SHARE, min=SHARE_RANGE[0],
            max=SHARE_RANGE[1],
            note="taken when it starts, after its weights, and kept until it stops, used or not; LMDeploy's own "
                 "default is 80"),
    Setting("tools", "Tool call parser", "choice", choices=PARSERS,
            note="the parser of the model's family; empty: no tool calls, and a coding harness cannot act (GPT-OSS "
                 "needs none)"),
)


def served_name(model: str) -> str:
    """The name a server answers to when none is set: its repository's (or folder's) last part, in lower case."""
    last = str(model or "").rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1].lower()
    cleaned = re.sub(r"[^a-z0-9._:-]+", "-", last).strip("-.") or "model"
    return cleaned[:128]


def answers_to(instance) -> str:
    """The name a local instance's server answers to (what a harness sends as its model): its own, else its model's
    last part in lower case; "" when it has no model."""
    served = str(instance.settings.get("served") or "").strip()
    model = str(instance.settings.get("model") or "").strip()
    return served or (served_name(model) if model else "")


def share_of(instance) -> int:
    """A local instance's cache share, in percent (the plugin's default when it sets none)."""
    share = instance.settings.get("share")
    return share if isinstance(share, int) and not isinstance(share, bool) else SHARE


# -- the model it is given ---------------------------------------------------------------------------------------------
def _folder_model(model: str) -> Path:
    path = Path(model)
    if not path.is_absolute():
        raise Refusal(f"{model[:120]} is no whole path: a model of yours is named by its folder's whole path")
    if mounts.foreign(path):
        raise Refusal(f"{path} is on another system's drive: a model there is that system's, served from there")
    if not path.is_dir():
        raise Refusal(f"{path} is not a folder here: name a folder that holds the model's files")
    return path


def model_folder(p, instance) -> Path:
    """The folder `lmdeploy serve api_server` is given for a local instance: its repository's snapshot, downloaded
    here (a start is offline, and a folder that is there is never looked up at the Hub), or its own folder. Refusal in
    words when it cannot be served so."""
    model = str(instance.settings.get("model") or "").strip()
    if not model:
        raise Refusal(f"set {instance.name}'s model first (Change…): LMDeploy serves one model, named when it starts")
    if _absolute(model):
        return _folder_model(model)
    from . import models

    entry = models.cached(p, model, fresh=True)  # (a start is a click: the folder as it is now)
    if entry is None:
        raise Refusal(f"download {model} first (Models): a server here is started offline, so a start never "
                      "downloads")
    if entry.incomplete or not entry.snapshot:
        raise Refusal(f"{model}'s download is not whole ({entry.why or 'a download was stopped'}): download it "
                      "again (Models)")
    return entry.path / "snapshots" / entry.snapshot


def config_of(folder: Path) -> dict:
    """A model's config.json, read ({} when it is not there or not JSON): its architectures and its quantization."""
    try:
        said = json.loads((Path(folder) / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    return said if isinstance(said, dict) else {}


def architecture(config: Mapping) -> str:
    named = config.get("architectures") if isinstance(config, Mapping) else None
    if isinstance(named, list) and named and isinstance(named[0], str):
        return named[0]
    return ""


def quantization(config: Mapping) -> str:
    """How a model's weights are quantized, as its config.json says (its quantization_config's quant_method, or its
    format): "" when it says none."""
    q = config.get("quantization_config") if isinstance(config, Mapping) else None
    if not isinstance(q, Mapping):
        return ""
    return str(q.get("quant_method") or q.get("format") or "").strip().lower()[:40]


def turbomind_refusal(folder: Path, label: str) -> str:
    """"" when TurboMind serves the model in `folder`, else why not, in words: what LMDeploy's command would hand to
    its PyTorch engine (an architecture TurboMind does not serve, SmoothQuant's weights) is refused here before a
    start. A TurboMind workspace (a folder holding triton_models) is TurboMind's own."""
    if (Path(folder) / "triton_models").is_dir():
        return ""
    config = config_of(folder)
    if not config:
        return (f"{label} has no config.json that can be read: which architecture it is cannot be told, and LMDeploy "
                "would hand a model TurboMind does not serve to its PyTorch engine, which is never used here")
    arch = architecture(config)
    if arch not in TURBOMIND:
        return (f"{label} is {arch or 'of no architecture named'}, which LMDeploy's TurboMind does not serve: LMDeploy "
                "would hand it to its PyTorch engine, which is never used here (it holds a loopback port with no key "
                "in a second process). TurboMind serves Qwen 2 to 3.5, Llama, InternLM, InternVL, Intern-S1 and S2, "
                "GLM-4.7 Lite, GPT-OSS and Mixtral")
    if quantization(config) == "smooth_quant":
        return (f"{label} is quantized by SmoothQuant, which TurboMind does not serve: LMDeploy would hand it to its "
                "PyTorch engine, which is never used here")
    return ""


def tiktoken_folder(p) -> Path:
    return p.home / "tiktoken"


def tiktoken_refusal(p, folder: Path) -> str:
    """"" unless the model is GPT-OSS and its vocabulary is not in the plugin's own folder whole: openai-harmony
    would download it at the first request, and a server here reaches nothing by itself."""
    if architecture(config_of(folder)) != GPT_OSS:
        return ""
    vocab = tiktoken_folder(p) / TIKTOKEN
    try:
        digest = hashlib.sha256(vocab.read_bytes()).hexdigest()
    except OSError:
        digest = ""
    if digest == TIKTOKEN_SHA256:
        return ""
    return (f"GPT-OSS reads its vocabulary ({TIKTOKEN}) at its first request and would download it from OpenAI's "
            f"storage; a server here downloads nothing, so put that file in {tiktoken_folder(p)} first (its sha256 "
            f"{TIKTOKEN_SHA256[:16]}…, which openai-harmony checks too)"
            + (": the file there is not that one" if digest else ""))


def environment(p) -> dict[str, str]:
    """A server's variables: offline and away from every hub, Ray's usage statistics off, its folders the plugin's
    own, uvicorn's per-request lines off, its output unbuffered (its log shows its progress as it comes)."""
    cache = p.home / "cache"
    env = {"HF_HOME": str(p.home / "models"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
           "LMDEPLOY_USE_MODELSCOPE": "False", "LMDEPLOY_USE_OPENMIND_HUB": "False",  # only 'true' switches a hub on
           "LMDEPLOY_EXECUTOR_BACKEND": "uni", "RAY_USAGE_STATS_ENABLED": "0",
           "TIKTOKEN_ENCODINGS_BASE": str(tiktoken_folder(p)), "TIKTOKEN_RS_CACHE_DIR": str(cache / "tiktoken"),
           "TORCH_HOME": str(cache / "torch"), "TRITON_CACHE_DIR": str(cache / "triton"),
           "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"), "UVICORN_LOG_LEVEL": "warning",
           "PYTHONUNBUFFERED": "1"}
    if p.system == "linux":
        env["XDG_CACHE_HOME"] = str(cache)
    return env


def launch(p, instance, key: str = "") -> children.Launch:
    """A local instance's server as it is started (the module's words). Refusal in words: LMDeploy not installed, no
    model, a repository not downloaded here or not whole, a folder that is not there or lies on another system's
    drive, a model TurboMind does not serve, GPT-OSS without its vocabulary here, a share outside 10 to 95. `key` is
    never given: LMDeploy takes one only on its command line (KEYED is False)."""
    head = release.program(p)
    if not head:
        raise Refusal("LMDeploy is not installed here: Install… first")
    folder = model_folder(p, instance)
    label = str(instance.settings.get("model") or folder)
    for why in (turbomind_refusal(folder, label), tiktoken_refusal(p, folder)):
        if why:
            raise Refusal(why)
    settings = instance.settings
    served = answers_to(instance) or served_name(str(folder))
    if not SERVED.fullmatch(served):
        raise Refusal(f"{instance.name}'s name to answer to ({served[:60]!r}) holds a slash or a space: change it")
    share = share_of(instance)
    if not SHARE_RANGE[0] <= share <= SHARE_RANGE[1]:
        raise Refusal(f"{instance.name}'s cache share is {share}%: {SHARE_RANGE[0]} to {SHARE_RANGE[1]}")
    port = p.port_of(instance)
    argv = [*head, "serve", "api_server", str(folder), "--server-name", LOOPBACK, "--server-port", str(port),
            "--model-name", served, "--backend", "turbomind", "--tp", "1", "--cache-max-entry-count",
            f"{share / 100:.2f}"]
    context = settings.get("context")
    if isinstance(context, int) and not isinstance(context, bool):
        argv += ["--session-len", str(context)]
    form = str(settings.get("format") or "").strip()
    if form:
        if form not in FORMATS:
            raise Refusal(f"{instance.name}'s weights' format ({form[:40]!r}) is none LMDeploy names: change it")
        argv += ["--model-format", form]
    tools = str(settings.get("tools") or "").strip()
    if tools:
        if tools not in {c[0] for c in PARSERS}:
            raise Refusal(f"{instance.name}'s tool call parser ({tools[:40]!r}) is none LMDeploy {release.VERSION} "
                          "registers: change it")
        argv += ["--tool-call-parser", tools]
    argv += ["--allow-origins", f"http://{LOOPBACK}:{port}", "--allowed-media-domains", MEDIA,
             "--disable-fastapi-docs", "--disable-metrics", "--log-level", "WARNING", "--max-log-len", "0",
             "--distributed-executor-backend", "uni"]
    return children.Launch(tuple(argv), env=environment(p), cwd=str(p.home), drop=DROP)


def fell_back(p, instance) -> bool:
    """Whether a local instance's log says LMDeploy handed its model to the PyTorch engine since its last start (its
    log's end read, from the line the library's child writes at each start, "liminal: NAME started (pid N)": the log
    is appended to across starts, so an earlier run's warning is not this one's. A file, nothing run)."""
    path = p.log_path(instance.name)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - LOG_TAIL))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return False
    at = tail.rfind(f"liminal: {instance.name} started (pid")
    return FALLBACK.lower() in (tail[at:] if at >= 0 else tail).lower()


def start_words(p, instance) -> str:
    """What a start of `instance` does, after the base's words."""
    model = str(instance.settings.get("model") or "").strip() or "its model (none is set yet: Change… first)"
    return (f"It loads {model} at once with LMDeploy's TurboMind, then keeps {share_of(instance)}% of the GPU's free "
            "memory for its cache until it is stopped, used or not. It is started offline: nothing is asked of "
            "Hugging Face. It binds 127.0.0.1, although LMDeploy's own default is every interface, and no other web "
            "page can read it, although LMDeploy's own default lets every site. It opens its one port and no other "
            "(TurboMind on one GPU is one process). It checks no key: every program of this PC can use it.")


def _models(p, instance, rows) -> list[dict]:
    out = []
    local = instance.where == "local"
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping) or row.get("parent"):
            continue  # an adapter's entry names its base as its parent
        detail = str(row.get("root") or "")
        if local:
            detail = f"{answers_to(instance) or detail} · {share_of(instance)}% of the free GPU memory for its cache"
        out.append({"id": row.get("id"), "size": None, "context": None, "loaded": True,
                    "tools": bool(instance.settings.get("tools")) if local else None, "detail": detail})
    return out


def look(p, at, instance) -> Seen:
    """What an instance answers now: GET /v1/models (its model, owned_by 'lmdeploy': else it is not LMDeploy; 401:
    it wants its key), GET /health (healthy, initializing, sleeping; 503 unhealthy). A local one that runs and does not
    answer yet is said so; one whose log says it fell back to the PyTorch engine is stopped, and said so. Asked by the
    Models tab and a click, never by a status."""
    address = p.address(instance)
    local = instance.where == "local"
    if local and instance.name in p.running() and fell_back(p, instance):
        stopped = p.stop(instance.name)
        return Seen(False, f"{instance.name}: LMDeploy handed its model to its PyTorch engine, which is never used "
                           f"here, so it is stopped ({' '.join(str(stopped.get('said', '')).split())[:200]}). Serve a "
                           "model TurboMind serves", facts={"engine": "fell back to PyTorch: stopped"})
    try:
        listed = at.get("/v1/models")
    except EndpointError as e:
        if e.status == 0 and local and instance.name in p.running():
            return Seen(False, NOT_YET)
        if e.refused:
            return Seen(False, f"something answers at {address} and wants a key: a remote LMDeploy started with "
                               "--api-keys takes its key (Key…)")
        raise
    rows = listed.get("data") if isinstance(listed, Mapping) else None
    if not isinstance(rows, list) or not any(isinstance(r, Mapping) and r.get("owned_by") == OWNER for r in rows):
        return Seen(False, f"what answers at {address} is not LMDeploy: its /v1/models names no model of "
                           f"{OWNER!r}'s")
    found = _models(p, instance, rows)
    version = release.VERSION if local else ""
    facts = {"engine": "TurboMind" if local else "its own where it was started",
             "metrics": "off" if local else "its own setting where it was started"}
    health = at.answer("GET", "/health", ok=(200, 503))
    state = health.data.get("status") if isinstance(health.data, Mapping) else None
    names = ", ".join(str(m["id"]) for m in found if m.get("id")) or "no model"
    if health.status == 503 or state == "unhealthy":
        then = "Stop, then Start" if local else "it is started again where it runs"
        return Seen(False, f"LMDeploy answers at {address}, but it says it is unhealthy: {then}", version=version,
                    models=tuple(found), facts={**facts, "health": "unhealthy"})
    if state in ("initializing", "sleeping"):
        return Seen(False, f"LMDeploy answers at {address}: {state}, for {names}", version=version,
                    models=tuple(found), facts={**facts, "health": state})
    return Seen(True, f"LMDeploy answers at {address}: it serves {names}", version=version, models=tuple(found),
                facts={**facts, "health": str(state or "healthy")})


# -- what leaves this machine ------------------------------------------------------------------------------------------
def leaves(p) -> dict:
    """The table "What leaves this machine", and what reaches it: every way LMDeploy, hf and the install reach
    outside, what is switched off here, and the ports it opens. Words alone: nothing read, nothing run."""
    rows = [
        ["Its port", "127.0.0.1 alone, the instance's own",
         "LMDeploy binds every interface unless told; here it binds this machine's loopback address and opens no "
         "other port (TurboMind on one GPU is one process). It checks no key: every program of this PC can use it, "
         "including /sleep, /wakeup and /update_weights, which LMDeploy answers without one"],
        ["Web pages", "only its own address (--allow-origins)",
         "LMDeploy's own default lets every site a browser opens read it; its command line can narrow that to one "
         "origin, not turn it off"],
        ["Hugging Face", "asked by Look up and Download only",
         "a server is started offline (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1) on a folder downloaded here: no "
         "check for newer files, no telemetry. ModelScope and openMind's hub are off. Your token, where you keep one, "
         "goes with a lookup and a download alone"],
        ["Images in a request", "no public host (--allowed-media-domains localhost)",
         "a vision model fetches the images a request names from any public host unless told otherwise; a request may "
         "still name a file of this machine's, which LMDeploy reads as it is"],
        ["GPT-OSS's vocabulary", "read from this plugin's folder",
         f"openai-harmony downloads {TIKTOKEN} at a GPT-OSS server's first request unless it is given a folder; a "
         "GPT-OSS start is refused until the file is there"],
        ["Telemetry", "none found in LMDeploy", "C-L1 read its command, its server and its engines and found no "
                                                "telemetry or update check; Ray, which comes with it, is never "
                                                "started here, and its usage statistics are off besides"],
        ["Its API pages and metrics", "off (--disable-fastapi-docs, --disable-metrics)",
         "its /docs page would load scripts from the internet in a browser; its /metrics would be one more path"],
        ["Install", "on your click", "uv, LMDeploy and what it needs come from GitHub, Astral's mirror, PyPI and "
                                     "PyTorch's index; nothing is looked for by itself afterwards"],
    ]
    return {"title": "What leaves this machine", "columns": [{"title": "What"}, {"title": "Here"},
                                                            {"title": "What it means"}], "rows": rows}


def size_said(size) -> str:
    return size_words(size) if isinstance(size, int) and size > 0 else "not said"


# the calls this module makes, by card (apis.py's names)
CALLS = {
    "LMDeploy's api_server": ("GET /v1/models", "GET /health", "POST /v1/chat/completions"),
    "lmdeploy, the command": ("lmdeploy serve api_server",),
    "Start at sign-in": ("Register-ScheduledTask -AtLogOn", "Get-ScheduledTask | Get-ScheduledTaskInfo",
                         "Start-ScheduledTask / Stop-ScheduledTask", "Unregister-ScheduledTask",
                         "systemctl --user daemon-reload / enable", "systemctl --user show",
                         "systemctl --user start / stop", "systemctl --user disable --now"),
}
