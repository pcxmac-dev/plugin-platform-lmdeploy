# plugin-platform-lmdeploy

LMDeploy for Liminal's platform: its environment installed on this PC or this Linux (a WSL 2 distribution included),
a server per model started and stopped here by its TurboMind engine alone, on 127.0.0.1, its models looked up,
downloaded and deleted, and instances elsewhere reached by their address.

The user's words (2026-11-18): "top 4 inference engines, top 4 agentic harnesses (open source)", "both linux and
windows compatibility". Their answers of 2026-11-25: LMDeploy is the fourth engine, TurboMind only, never the silent
PyTorch fallback (no Responses API, so Codex CLI cannot use it); the ports a tool opens by itself are kept and said on
its page. Before them (2026-11-16): "all three plugins need to allow installation and configuration of their
respective code"; "also remote connection to instances if possible"; "the plugins manage the services, the app can be
started @ boot like a service, ports can be configured via the plugin, the plugin should report conflicts". And
(2026-11-17): "all plugins are middleware", "even for other plugins", "and then the app object code".

## What it is

A machine plugin of the platform (plugin API 2.2) built on liminal-libs 1.66.0's `ModelServer`
(`liminal_libs.machines.modelserver`), drawn under the platform's AI heading in the Inference group
(`SECTION = "ai"`, `GROUP = "inference"`, `order = 513`). Its icon is LMDeploy's own mark, drawn anew in the set's
line (`ICON = "lmdeploy"`, in liminal-libs' icons.js).

As middleware it answers the standard interface every inference plugin answers (its instances, their starts and
stops, their ports and conflicts, their start at boot, their models, a prompt to try, and `serving` and `models_of`,
which the harness plugins ask through the platform), and adds four hooks of its own: `find`, `clear_found`, `pull`
and `delete_model`. Its rows say it speaks `chat` and `completions` (`SPEAKS`): LMDeploy 0.18.0 has no OpenAI
Responses API. What it sits on (its External API) is LMDeploy's api_server and its command, uv, Hugging Face's hf and
Hub, the package indexes its install reads, nvidia-smi, and the scheduled tasks and systemd user units a start at boot
is.

**Valid wherever it serves anything**, which is everywhere: an instance elsewhere can always be added by its address.
What a system lacks is a fact and a refusal in words, never a reason to be invalid.

## Where it runs

- **Windows x64**: LMDeploy 0.18.0's win_amd64 wheel, TurboMind on one GPU, with kernels for sm_70, 75, 86, 89 and 120
  (an RTX 50-series card is sm_120; an A100, H100 or B200 has none). Its CUDA 12 libraries come from PyTorch's own
  cu128 wheel: no CUDA toolkit is installed for the whole system.
- **Linux x86-64 with glibc 2.28 or newer**, a WSL 2 distribution included: its manylinux_2_28 wheel, with NVIDIA's
  CUDA 12 runtime, cuBLAS, cuRAND and NCCL from PyPI. WSL 1, musl (Alpine) and an older glibc are refused in words.
- **Elsewhere** (ARM64 on either system): nothing installs; instances elsewhere are added by their address.
- NVIDIA's driver: CUDA 12.8 Update 1 needs 572.61 or newer on Windows, 570.124.06 or newer on Linux (under WSL it is
  Windows' driver). The install reads it with nvidia-smi inside its job and refuses an older one.

## Install, update, uninstall

Install… is a job, asked twice (its size, hosts and licence first, then the licence accepted), and cancellable. It
needs no administrator and runs no sudo:

1. The GPU, its driver and the room (20 GB free) are looked at first; each refusal is said in words.
2. uv 0.13.8 is fetched from its GitHub release (its Windows .zip or its Linux .tar.gz) and checked against its pinned
   size and sha256, then unpacked into home/app/0.18.0/uv/bin.
3. `uv venv --python 3.12 --managed-python --no-project --no-config` makes the environment on uv's own CPython 3.12.16,
   with nothing of the user's uv, pip or Python configuration read.
4. LMDeploy's wheel (from PyPI's file host, by size and sha256) and PyTorch 2.11.0's torch and torchvision wheels for
   CUDA 12.8 (from PyTorch's own index, by sha256 alone: its index lists no sizes) are fetched and checked.
5. `uv pip install --no-build --torch-backend=cu128` installs the three wheels, the rest of PyTorch's family from its
   cu128 index and everything else from PyPI, as wheels alone. PyTorch's index is never an extra index
   (`--extra-index-url` would let it serve PyPI's projects too). No extra of LMDeploy's is asked for.
6. A windowless check, 120 s at most: the environment's Python imports LMDeploy 0.18.0 and loads TurboMind's native
   module. An environment whose TurboMind does not load is never put in use: LMDeploy's command would hand every model
   to its PyTorch engine.
7. The install is put in use (app/CURRENT), and its downloads are let go.

What is checked: uv's archive and the three wheels, against the sha256 pinned here. What uv takes besides them from
PyPI and PyTorch's index comes over https as those indexes give it, and no sum pinned here checks it: the install
says so before it starts and in its panel. Its licence is LMDeploy's Apache-2.0; PyTorch (BSD-3-Clause), NVIDIA's CUDA
libraries (NVIDIA's own terms), Ray, Hugging Face's libraries and uv carry their own, which nothing here asks you to
accept. Each model's licence is said before it is downloaded.

Uninstall… removes the install, its downloads and its caches (home/cache); its models, its logs and the instances
stay. An `lmdeploy` of your own on PATH is used as it is, its version read from its files, and never updated,
replaced or removed from here.

## Instances and servers

A local instance is one server serving one model, a windowless child the plugin owns (its log in home/logs), or its
scheduled task's or systemd user unit's when it starts at boot. Its settings: the model (a repository downloaded here,
or a folder's whole path), the name it answers to, its weights' format (empty: LMDeploy reads it from the model), its
session length, its cache share (70% of the GPU's free memory after its weights; LMDeploy's own default is 80) and
its tool call parser (one of the thirteen LMDeploy 0.18.0 registers). It is started

```
<lmdeploy> serve api_server <the snapshot's folder> --server-name 127.0.0.1 --server-port <port>
           --model-name <name> --backend turbomind --tp 1 --cache-max-entry-count 0.70
           [--session-len N] [--model-format F] [--tool-call-parser P]
           --allow-origins http://127.0.0.1:<port> --allowed-media-domains localhost
           --disable-fastapi-docs --disable-metrics --log-level WARNING --max-log-len 0
           --distributed-executor-backend uni
```

- **127.0.0.1 always.** LMDeploy's own default is every interface.
- **No key.** LMDeploy takes keys on its command line alone (`--api-keys`), where every program of this machine could
  read them; this plugin writes no key on a command line, so its servers check none (`KEYED = False`), Key… and
  Share… are disabled for a local instance with why, and no local server is shared beyond this machine from here.
  Every program of this PC can use it, including /sleep, /wakeup and /update_weights, which LMDeploy answers without
  a key: its page says so.
- **Origins narrowed.** LMDeploy's own default lets every web page a browser opens read it; its command line cannot
  turn cross-origin answers off, only narrow them to one origin, here its own address.
- **TurboMind alone.** LMDeploy's command hands a model TurboMind does not serve to its PyTorch engine with a warning
  alone, and that engine always starts a second process holding an unauthenticated loopback port. So a model whose
  config.json names an architecture outside TurboMind's nineteen, or that SmoothQuant quantized, or that has no
  config.json, is refused before the start; Serve it… is disabled for it on the Models tab. Should LMDeploy fall back
  all the same, its log's warning ("Fallback to pytorch engine") is read at every draw of the page (a file read),
  before another plugin is told what it serves and when the instance is looked at, and the server is stopped, said
  as a problem. `--distributed-executor-backend uni` keeps Ray out even then.
- **One port.** TurboMind on one GPU is one process, and its only listening socket is the instance's port.
- **Offline.** `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, the snapshot's whole path given, ModelScope and
  openMind's hub off: a start never downloads. A vision model fetches no image from any public host
  (`--allowed-media-domains localhost`); a request may still name a file of this machine's, which LMDeploy reads.
- **GPT-OSS** reads its vocabulary (o200k_base.tiktoken) from the plugin's own folder alone
  (`TIKTOKEN_ENCODINGS_BASE`): a GPT-OSS start is refused until that file is there with the sha256 openai-harmony
  checks. The plugin does not fetch it.
- **Nothing logged of a prompt.** `--log-level WARNING` and `--max-log-len 0` (prompts and answers are logged only
  below INFO), `UVICORN_LOG_LEVEL=warning` (no per-request lines).
- **Its environment** is the platform's without LMDeploy's, the Hub's, Transformers', ModelScope's, PyTorch's,
  Triton's, Ray's, tiktoken's, uvicorn's, TLS's and Python's own variables and every pair of Liminal's, with its own
  laid over it; its caches (torch, Triton, inductor, tiktoken) are in home/cache.

A found instance (on Windows, an LMDeploy inside WSL answering at 127.0.0.1:23333 through WSL's relay) and a remote one
are asked, never started or stopped from here; a remote one started with `--api-keys` takes its key (Key…), a secret
pair of its own. LMDeploy is told apart from other servers by `owned_by: "lmdeploy"` in its `/v1/models`.

## Models

The Models tab, on the library's `machines/hub.py`: Find a model asks the Hub's API for a model's card (its licence),
its gate, its files and its exact size before anything is downloaded (no install needed); Download… runs the
environment's own `hf download <repo> --quiet` into home/models (`HF_HOME`; the Hub's shared store off, so a delete
frees what it says), its progress from the folder's bytes, the user's Hugging Face token only where one is kept (as
`HF_TOKEN`, never on a command line), Cancel keeping what came; Downloaded here lists each model with what its
config.json says (its architecture, whether TurboMind serves it, its weights' format: AWQ, GPTQ, compressed-tensors,
FP8 or MXFP4); Delete… removes a model's folder and its lock files, refused while a running server serves it.

## What leaves this machine

Its Main says it, row by row: its port (127.0.0.1 alone, and no other), web pages (its own address alone), Hugging
Face (asked by Look up and Download only), images in a request (no public host), GPT-OSS's vocabulary (from the
plugin's folder), telemetry (none found in LMDeploy; Ray never started, its statistics off besides), its API pages and
metrics (off), and the install (on your click alone).

## Its interface (Internal API)

No reads of its own beside the bases'. Its own hooks are asked only from its page (`PAGE_ONLY_HOOKS`): `pull` and
`delete_model`, and since 0.1.1 `find` and `clear_found` (the user, 2026-11-26, of reads made with a key: "Page-only").
`find` asks huggingface.co about a name of the asker's choosing with your Hugging Face token where one is kept, and
`clear_found` empties the lookups the Models tab keeps. Another plugin's ask and an ask that names no one are refused
(403) before anything is sent. The model server's own changes are the library's to say. What it keeps through the login:
its instances (`LIMINAL_AI_LMDEPLOY_INSTANCES`) and a remote instance's key (`LIMINAL_AI_LMDEPLOY_<INSTANCE>_KEY`); its
one key of its own is the optional Hugging Face token (`LIMINAL_AI_LMDEPLOY_HF_TOKEN`). Its `NEEDS` rows (architecture
x86-64, an NVIDIA GPU and its driver, glibc 2.28, WSL 2, 20 GiB, the internet), each a part, are written in the
getattr idiom and read once the library has a needs module.

## Its API

Eight cards (`apis.py`): LMDeploy's api_server, lmdeploy the command, uv, hf, the Hugging Face Hub API, the package
indexes, nvidia-smi and Start at sign-in. Each names what the plugin does with each part, each call it makes and the
notable ones it does not (/v1/responses, /update_weights, /metrics among them).

## From the command line

```
python -m plugin_platform_lmdeploy status          its Main, as JSON (files and kept readings: nothing run)
python -m plugin_platform_lmdeploy serving         what it tells the other AI plugins: its instances (no key)
python -m plugin_platform_lmdeploy models NAME     an instance asked now for its models (over http, a few seconds)
python -m plugin_platform_lmdeploy launch NAME     a local instance's command as lines to paste (no key in it)
```

## Where it is built

A local repository of its own, built in a scratch folder for item 19 of the Monday list; moving it into incoming/,
making its GitHub repository and pushing it are each the user's yes. Its CI workflow runs once a repository exists.

## Tests

No test starts a process, opens a socket, resolves a name or downloads anything: an autouse guard (`tests/conftest.py`)
fails a test that does, and the plugin is driven with stand-ins for its environment, its system, its folder, its
clock, its lookups, its socket table, its http, its downloads, its commands, nvidia-smi, the boot registrations and
the platform, so the suite reads alike on Windows, inside WSL and on a Linux of its own. `tests/data` holds the Hub's
answers (Qwen/Qwen3-0.6B) and LMDeploy's in the shapes its sources give (C-L1), written from them, not recorded.

- `test_contract.py`: the class contract, the AI section and the Inference group, the hook set, the jobs, KEYED and
  SPEAKS, the icon, the version and the floor, the Internal API's words, no line over 120 characters.
- `test_apis.py`: the cards sound; the calls they say it makes exactly its modules' `CALLS`; every host it names on a
  card or pinned; what only a real run shows listed as such here; facts that ask nothing.
- `test_release.py`: the pins as C-L1 read them; what is offered on each system and processor and why not; what is
  installed; the install on both systems over stand-ins; every refusal and failure, a TurboMind that does not load
  among them; a cancel; the uninstall.
- `test_server.py`: the settings; the command line and the environment on both systems; no key anywhere; every
  refusal, a model TurboMind does not serve among them; a start; what the harnesses are told; what an instance
  answers; a server that fell back to PyTorch, stopped.
- `test_models.py`: lookups, downloads, the models folder and what each model is, deletes, what fits.
- `test_page.py`: Main asking and running nothing; its two tables on each system; the Models tab; every form its
  hook's arguments; no token in any answer; the command line.
- `test_who_asks.py`: its two changes, its lookup and the lookups forgotten asked only from its page, nothing sent
  to the Hub for another plugin or for no one.
- `test_needs.py`: its NEEDS rows, read through getattr alone.

```
PYTHONPATH=src python -m pytest -q -p no:cacheprovider
```

## Not yet confirmed by a real run

Nothing was installed, started or downloaded while this plugin was built. What only a first real run shows:

- That the win_amd64 wheel's TurboMind's native module loads its CUDA 12 runtime, cuBLAS and cuRAND from torch
  2.11.0+cu128's own DLLs with no CUDA toolkit installed (C-L1 3d: likely, the module's import table not read), and
  that the import check proves it.
- That `uv pip install --no-build --torch-backend=cu128` resolves LMDeploy 0.18.0's requirements on both systems as
  wheels alone, and with which openai, transformers and xgrammar (the newest of each postdate LMDeploy 0.18.0); the
  sizes and the time each step takes (the download's size said is an estimate; torch's wheel size is not listed).
- That `--allow-origins http://127.0.0.1:<port>` answers no other origin, and that `--allowed-media-domains localhost`
  refuses every remote host without a lookup (C-L1's inference, not run).
- That LMDeploy's warning reads "Fallback to pytorch engine" in its log at the default WARNING level, so the stop
  after a fallback is ever reached; and that the architecture check before a start matches TurboMind's own.
- That a server's only listening socket is its port with TurboMind on one GPU, on Windows and on Linux.
- That a server serves a model from the snapshot's folder offline (`HF_HUB_OFFLINE=1`), on this PC's RTX 5090, with a
  70% cache share; how long a first start loads.
- That GPT-OSS reads o200k_base.tiktoken from `TIKTOKEN_ENCODINGS_BASE` and asks nothing outside.
- That Stop ends LMDeploy's one process whole on Windows and on Linux, and that the scheduled task and the user unit
  start it as at a click.

## Decisions it rests on

The plan's LMDeploy sections (D1, D2) and the pre-build checks' C-L1, with the user's answers of 2026-11-25. Where
this plugin differs from the plan: there is no `backend` setting (TurboMind alone, the user's answer, and PyTorch's
engine breaks the one-port rule on Linux too); the weights' format is empty by default (LMDeploy reads it) rather than
`hf`; the model is given as the snapshot's whole path, never the repository's name, so no hub is asked at a start; a
model TurboMind does not serve is refused by its config.json before a start; a lookup reads the Hub's API for the
size (as vLLM's does) rather than hf's dry run, so it needs no install; the import check is given 120 s rather than
60; GPT-OSS's vocabulary is not fetched by this plugin.
