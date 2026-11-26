"""What the LMDeploy plugin attaches to (plugin API 2.2; the user, 2026-11-11: "each plugin should reflect the the API
its attached to, and present links that provide documentation ... it should also expose more of what the API is
about"; 2026-11-16: "for all apps, and all plugins, there should be an external api tab"): LMDeploy's api_server and
its command, uv, Hugging Face's hf command and its Hub's API, the package indexes the install reads, nvidia-smi and
the start at sign-in (a scheduled task on Windows, a systemd user unit on Linux). Each part is named with what the
plugin does with it, each call it makes with what it makes it for, and the notable calls it does not make. The facts
are LMDeploy 0.18.0's, read in its sources at the tag v0.18.0 and on its documentation's pages on 2026-11-18 by the
pre-build checks (C-L1) and the plan (`VERIFIED`); liminal_libs.sources.apis.dead_links checks the pages with
LIMINAL_VENDOR_PAGES=1."""
from __future__ import annotations

from liminal_libs.sources.apis import Api, Area, Call, Link

from . import release

VERIFIED = "2026-11-18"
SERVER = "LMDeploy's api_server"
COMMAND = "lmdeploy, the command"
UV = "uv"
HF = "hf, Hugging Face's command"
HUB = "Hugging Face Hub API"
INDEXES = "The package indexes"
NVIDIA = "nvidia-smi"
BOOT = "Start at sign-in"

DOC = "https://lmdeploy.readthedocs.io/en/latest/"
REPO = "https://github.com/InternLM/lmdeploy"
TAG = REPO + "/blob/v0.18.0/"
HFDOC = "https://huggingface.co/docs/huggingface_hub/en/"
HUBDOC = "https://huggingface.co/docs/hub/"
UVDOC = "https://docs.astral.sh/uv/"
SYSTEMD = "https://www.freedesktop.org/software/systemd/man/latest/"
LEARN = "https://learn.microsoft.com/en-us/"
TASKS = LEARN + "powershell/module/scheduledtasks/"
WS = "?view=windowsserver2025-ps"

LMDEPLOY_SERVER = Api(
    name=SERVER, maker="InternLM (Shanghai AI Laboratory)", kind="rest",
    about=("The HTTP server `lmdeploy serve api_server` runs: OpenAI's API (models, chat and text completions, "
           "responses), Anthropic's Messages, and LMDeploy's own generate, health, sleep and weight paths. One server "
           "serves one model, named when it starts: it has no call that downloads, deletes, loads or unloads one. The "
           "plugin asks an instance what it is and what it serves while its Models tab is shown, and sends one prompt "
           "on Try it…; the harnesses ask it through its OpenAI-compatible /v1. LMDeploy tells itself apart by "
           "owned_by 'lmdeploy' in its /v1/models."),
    base="http://127.0.0.1:23333 (an instance's own address: this machine's port, or another machine's address)",
    version="LMDeploy 0.18.0's api_server (2026-09-28)",
    auth=("none here: LMDeploy checks a key only where it was started with --api-keys (a bearer, compared by its "
          "SHA-256), and this plugin never writes a key on a command line; /health, /docs and /nodes never ask one"),
    limits="one model a server; TurboMind serves 19 architectures, on one GPU on Windows",
    costs="none: LMDeploy is free software (Apache-2.0)",
    links=(
        Link("OpenAI Compatible Server", DOC + "llm/api_server.html", "reference"),
        Link("Tools calling", DOC + "llm/api_server_tools.html", "guide"),
        Link("Supported models", DOC + "supported_models/supported_models.html", "reference"),
        Link("The api_server's source at 0.18.0", TAG + "lmdeploy/serve/openai/api_server.py", "source"),
        Link("LMDeploy 0.18.0's release", REPO + "/releases/tag/v0.18.0", "changelog"),
        Link("InternLM/lmdeploy", REPO, "source"),
    ),
    areas=(
        Area("Models", "GET /v1/models: the model it serves, owned_by 'lmdeploy'", DOC + "llm/api_server.html",
             covered="whether what answers is LMDeploy, and its model, on the Models tab and for the harnesses"),
        Area("Chat completions", "OpenAI's chat API, streamed or not, tools when a parser was named at its start",
             DOC + "llm/api_server.html", covered="Try it…: one prompt, 512 tokens at most, not streamed"),
        Area("Health", "GET /health: healthy, initializing, sleeping, or unhealthy (503), without a key",
             TAG + "lmdeploy/serve/core/health.py", covered="whether its engine serves"),
        Area("Completions", "OpenAI's text completions, and LMDeploy's own /generate", DOC + "llm/api_server.html"),
        Area("Responses", "OpenAI's Responses API, POST /v1/responses alone (no stored responses)",
             DOC + "llm/api_server.html"),
        Area("Anthropic Messages", "Anthropic's Messages API and its models list, on the same port",
             DOC + "llm/api_server.html"),
        Area("Sleep and weights", "/sleep, /wakeup, /update_weights and the PyTorch engine's distributed paths, "
                                  "answered without a key where none was set",
             TAG + "lmdeploy/serve/openai/api_server.py"),
        Area("Metrics", "GET /metrics: Prometheus' text, names starting lmdeploy:", DOC + "llm/api_server.html",
             covered="switched off at a start here (--disable-metrics)"),
        Area("Its API pages", "/ and /docs: FastAPI's Swagger page, which loads scripts from the internet in a browser",
             DOC + "llm/api_server.html", covered="switched off at a start here (--disable-fastapi-docs)"),
    ),
    calls=(
        Call("GET /v1/models", "the model it serves, as an OpenAI list (owned_by 'lmdeploy')", "Models",
             used="whether what answers is LMDeploy, and its model (the Models tab; models_of for the harnesses)",
             docs=DOC + "llm/api_server.html"),
        Call("GET /health", "its engine's state: 200, or 503 when unhealthy", "Health",
             used="whether it serves yet (the Models tab)", docs=TAG + "lmdeploy/serve/core/health.py"),
        Call("POST /v1/chat/completions", "one chat answered", "Chat completions",
             used="Try it…: one prompt, 512 tokens and 4,000 characters at most, waited for 120 s",
             docs=DOC + "llm/api_server.html"),
        Call("POST /v1/completions", "a text completion", "Completions", docs=DOC + "llm/api_server.html"),
        Call("POST /generate", "LMDeploy's own generation", "Completions", docs=DOC + "llm/api_server.html"),
        Call("POST /v1/responses", "a response, OpenAI's Responses way", "Responses", docs=DOC + "llm/api_server.html"),
        Call("POST /v1/messages", "a message, Anthropic's way", "Anthropic Messages", docs=DOC + "llm/api_server.html"),
        Call("GET /anthropic/v1/models", "its models, Anthropic's way", "Anthropic Messages",
             docs=DOC + "llm/api_server.html"),
        Call("POST /sleep", "the engine put to sleep, without a key where none was set", "Sleep and weights",
             docs=TAG + "lmdeploy/serve/openai/api_server.py", writes=True),
        Call("POST /update_weights", "the model's weights replaced, without a key where none was set",
             "Sleep and weights", docs=TAG + "lmdeploy/serve/openai/api_server.py", writes=True),
        Call("GET /metrics", "its counters, for Prometheus", "Metrics", docs=DOC + "llm/api_server.html"),
    ),
    verified=VERIFIED)

LMDEPLOY_COMMAND = Api(
    name=COMMAND, maker="InternLM (Shanghai AI Laboratory)", kind="command",
    about=("The lmdeploy command of LMDeploy's Python package: `lmdeploy serve api_server` runs the server for one "
           "model, with TurboMind (its CUDA engine, one process on one GPU) or its PyTorch engine (always a second "
           "process with a loopback port of its own). Its command hands a model TurboMind does not serve to PyTorch "
           "with a warning alone; this plugin refuses such a model before a start and stops a server whose log says "
           "it fell back. Its other commands (chat, lite, convert, check_env) are not run from here; the install's "
           "check imports the package with the environment's own Python."),
    base=("lmdeploy (home/app/0.18.0/venv, this plugin's: Scripts\\lmdeploy.exe on Windows, bin/lmdeploy on Linux; "
          "or an lmdeploy of yours on PATH, used as it is)"),
    version="LMDeploy 0.18.0 (PyPI, 2026-09-28): CPython 3.10 to 3.14 wheels for win_amd64 and manylinux_2_28_x86_64",
    auth="none of its own: a key would be --api-keys on its command line, which this plugin never writes",
    limits="TurboMind on Windows: one GPU, kernels for sm_70, 75, 86, 89 and 120 alone",
    costs="none: free software (Apache-2.0)",
    links=(
        Link("The command line", DOC + "api/cli.html", "reference"),
        Link("Installation", DOC + "get_started/installation.html", "docs"),
        Link("Supported models", DOC + "supported_models/supported_models.html", "reference"),
        Link("cli/serve.py at 0.18.0", TAG + "lmdeploy/cli/serve.py", "source"),
        Link("lmdeploy 0.18.0 on PyPI", "https://pypi.org/project/lmdeploy/0.18.0/", "tool"),
    ),
    areas=(
        Area("serve api_server", "a server for one model: its address, port, name, engine, cache share, session "
                                 "length, format, tool parser, origins, media hosts, logs and metrics",
             DOC + "api/cli.html",
             covered=("a local instance's server: 127.0.0.1, its port, its name, TurboMind on one GPU, its share, "
                      "length, format and parser, origins narrowed to its own address, no media host, its API pages "
                      "and metrics off, no prompt logged, offline")),
        Area("The import check", "the environment's Python imports the package and TurboMind's native module",
             DOC + "get_started/installation.html", covered="once, at the end of an install"),
        Area("chat", "a model talked to in a terminal", DOC + "api/cli.html"),
        Area("lite", "a model quantized (AWQ, SmoothQuant, GPTQ)", DOC + "api/cli.html"),
        Area("serve proxy", "a proxy over several servers, each registering with it", DOC + "api/cli.html"),
        Area("check_env", "what the environment holds, nvidia-smi among it", DOC + "api/cli.html"),
    ),
    calls=(
        Call("lmdeploy serve api_server", "MODEL --server-name 127.0.0.1 --server-port N --backend turbomind ...",
             "serve api_server", used="Start of a local instance (and its task or unit at boot)",
             docs=DOC + "api/cli.html"),
        Call("python -I -c (the import check)", "import lmdeploy; from lmdeploy.turbomind import _turbomind",
             "The import check", used="the install's last step, windowless, 120 s at most",
             docs=DOC + "get_started/installation.html"),
        Call("lmdeploy chat", "a model in a terminal", "chat", docs=DOC + "api/cli.html"),
        Call("lmdeploy lite auto_awq", "a model quantized to 4 bits", "lite", docs=DOC + "api/cli.html", writes=True),
        Call("lmdeploy serve proxy", "a proxy server", "serve proxy", docs=DOC + "api/cli.html"),
        Call("lmdeploy check_env", "the environment said", "check_env", docs=DOC + "api/cli.html"),
    ),
    verified=VERIFIED)

UV_TOOL = Api(
    name=UV, maker="Astral", kind="command",
    about=("uv, Astral's Python package and environment manager. The plugin brings its own pinned copy from its "
           "release archive (checked against its sha256) and runs it with everything it keeps inside the install's "
           "folder: a CPython 3.12.16 of its own, the environment LMDeploy is installed into, and its cache under the "
           "plugin's downloads/ while the install runs. No configuration file of yours or of a folder is read, and "
           "your own UV_* variables are taken out."),
    base="uv (home/app/0.18.0/uv/bin, from its archive for this system)",
    version=f"uv {release.UV} (2026-11-12): --torch-backend is a preview switch of it",
    auth="none: PyPI and PyTorch's index are read anonymously",
    limits="", costs="none: free software (MIT or Apache-2.0)",
    links=(
        Link("Installing uv", UVDOC + "getting-started/installation/", "docs"),
        Link("uv's command line", UVDOC + "reference/cli/", "reference"),
        Link("Environment variables", UVDOC + "reference/environment/", "reference"),
        Link("Using uv with PyTorch", UVDOC + "guides/integration/pytorch/", "guide"),
        Link("Package indexes", UVDOC + "concepts/indexes/", "docs"),
        Link(f"uv {release.UV}", f"https://github.com/astral-sh/uv/releases/tag/{release.UV}", "changelog"),
        Link("astral-sh/uv", "https://github.com/astral-sh/uv", "source"),
    ),
    areas=(
        Area("Releases", "uv's own archives, one per system, each with its .sha256",
             UVDOC + "getting-started/installation/",
             covered="its Windows .zip and its Linux .tar.gz, pinned by their size and sha256"),
        Area("Environments", "uv venv: an environment on a Python asked for", UVDOC + "reference/cli/",
             covered="the environment LMDeploy runs from (--python 3.12 --managed-python)"),
        Area("Packages", "uv pip install: packages into an environment, PyTorch's build chosen by --torch-backend",
             UVDOC + "guides/integration/pytorch/",
             covered="LMDeploy's, torch's and torchvision's wheels, --no-build, --torch-backend=cu128"),
        Area("Managed Python", "Pythons uv downloads (Astral's mirror first, GitHub after a failure) and checks",
             UVDOC + "reference/environment/", covered="CPython 3.12.16, kept inside the install's folder"),
        Area("Projects and locks", "uv lock, sync and run, over a project's pyproject.toml", UVDOC + "reference/cli/"),
    ),
    calls=(
        Call("GET uv's archive", "uv's archive for this system from its GitHub release", "Releases",
             used="the install's first download, checked against its pinned size and sha256",
             docs=UVDOC + "getting-started/installation/"),
        Call("uv venv", "--python 3.12 --managed-python --no-project --no-config FOLDER", "Environments",
             used="the install's environment", docs=UVDOC + "reference/cli/", writes=True),
        Call("uv pip install", "--python PYTHON --no-config --no-build --torch-backend=cu128 THE-THREE-WHEELS",
             "Packages", used="LMDeploy and what it needs into the environment", docs=UVDOC + "reference/cli/",
             writes=True),
        Call("uv pip compile", "a requirements file with every package's sums", "Projects and locks",
             docs=UVDOC + "reference/cli/"),
    ),
    verified=VERIFIED)

HF_COMMAND = Api(
    name=HF, maker="Hugging Face", kind="command",
    about=("hf, the command of the huggingface_hub library that comes into LMDeploy's environment with transformers. "
           "The plugin downloads a model with it into its own models folder, told its folder (HF_HOME), its telemetry, "
           "its update check and its progress bars off, the Hub's shared store of files off, and the user's token only "
           "where one is kept."),
    base="hf (beside lmdeploy in its environment)",
    version="huggingface_hub as uv resolves transformers' requirement (1.31.0 or newer at transformers 5.23.0)",
    auth="the user's Hugging Face token as HF_TOKEN, where one is kept, for a gated or private model; never --token",
    limits="the Hub's rate limits (its Rate limits page)", costs="none for public models",
    links=(
        Link("Command Line Interface", HFDOC + "guides/cli", "guide"),
        Link("hf: its reference", HFDOC + "package_reference/cli", "reference"),
        Link("Understand caching", HFDOC + "guides/manage-cache", "docs"),
        Link("Environment variables", HFDOC + "package_reference/environment_variables", "reference"),
        Link("huggingface/huggingface_hub", "https://github.com/huggingface/huggingface_hub", "source"),
    ),
    areas=(
        Area("Downloads", "hf download: a repository's files into the cache", HFDOC + "guides/cli",
             covered="Download… of a model, quiet, into this plugin's models folder"),
        Area("The cache", "models--org--name folders of blobs, snapshots and refs", HFDOC + "guides/manage-cache",
             covered="read by this plugin to list what is downloaded, and deleted from it by Delete…"),
        Area("Environment variables", "HF_HOME, HF_TOKEN, HF_HUB_OFFLINE and the switches",
             HFDOC + "package_reference/environment_variables",
             covered=("HF_HOME, and HF_HUB_DISABLE_TELEMETRY, _UPDATE_CHECK, _SHARED_BLOBS and _PROGRESS_BARS set to "
                      "1; HF_HUB_OFFLINE=1 for a server")),
        Area("Cache tools", "hf cache ls, rm, prune and verify", HFDOC + "guides/manage-cache"),
        Area("Sign-in", "hf auth login: a token stored on the machine", HFDOC + "guides/cli"),
    ),
    calls=(
        Call("hf download", "REPO --quiet: a model's files into HF_HOME's hub", "Downloads",
             used="Download… on the Models tab, as a job", docs=HFDOC + "guides/cli", writes=True),
        Call("hf download --dry-run", "what a download would take, rounded", "Downloads",
             docs=HFDOC + "package_reference/cli"),
        Call("hf cache rm", "a repository deleted from the cache", "Cache tools", docs=HFDOC + "guides/manage-cache",
             writes=True),
        Call("hf auth login", "a token stored on the machine (never from here)", "Sign-in", docs=HFDOC + "guides/cli",
             writes=True),
    ),
    verified=VERIFIED)

HUB_API = Api(
    name=HUB, maker="Hugging Face", kind="rest",
    about=("The Hugging Face Hub's own API: its models, their files and their cards. The plugin asks it, when a model "
           "is looked up, for the model's card data (its licence), its gate and its files, and for its exact size, "
           "before anything is downloaded. A download's files it sends on to the storage it names per request, which "
           "hf reaches; the plugin itself asks huggingface.co alone."),
    base=release.HUB,
    version="the Hub's API as its OpenAPI specification gives it; the model information path is the client library's",
    auth=("none for a public model; the user's token as a bearer, where one is kept (Settings), for a gated or "
          "private one"),
    limits="per 5-minute window, by plan (its Rate limits page); 429 past them",
    costs="none to read public models; its own plans for accounts",
    links=(
        Link("Hub API Endpoints", HUBDOC + "api", "docs"),
        Link("OpenAPI Playground", "https://huggingface.co/spaces/huggingface/openapi", "reference"),
        Link("Model cards", HUBDOC + "model-cards", "docs"),
        Link("Licenses", HUBDOC + "repositories-licenses", "reference"),
        Link("Gated models", HUBDOC + "models-gated", "guide"),
        Link("User access tokens", HUBDOC + "security-tokens", "auth"),
        Link("Rate limits", HUBDOC + "rate-limits", "limits"),
    ),
    areas=(
        Area("Model information", "GET /api/models/{repo}: its card's data, its gate, its files",
             HFDOC + "package_reference/hf_api", covered="a model's licence, whether it is gated, how many files"),
        Area("Folder sizes", "GET /api/models/{repo}/treesize/{rev}: the bytes of all files under a path",
             "https://huggingface.co/spaces/huggingface/openapi", covered="a model's exact size before a download"),
        Area("Resolve", "GET /{repo}/resolve/{rev}/{file}: a file's content, sent on to its storage",
             HUBDOC + "rate-limits"),
        Area("Search", "GET /api/models: models by what they are", "https://huggingface.co/spaces/huggingface/openapi"),
    ),
    calls=(
        Call("GET /api/models/{repo}", "?expand=cardData&expand=gated&expand=siblings: its card, gate and files",
             "Model information", used="Look up: a model's licence, gate and files",
             docs=HFDOC + "package_reference/hf_api"),
        Call("GET /api/models/{repo}/treesize/main", "{\"path\": \"/\", \"size\": N}: the bytes of its files",
             "Folder sizes", used="Look up: a model's exact size",
             docs="https://huggingface.co/spaces/huggingface/openapi"),
        Call("GET /{repo}/resolve/main/{file}", "a file's content (what hf downloads through)", "Resolve",
             docs=HUBDOC + "rate-limits"),
        Call("GET /api/models", "a search of the Hub's models", "Search",
             docs="https://huggingface.co/spaces/huggingface/openapi"),
    ),
    verified=VERIFIED)

PACKAGE_INDEXES = Api(
    name=INDEXES, maker="PyPI (the Python Software Foundation) and PyTorch", kind="feed",
    about=("The indexes LMDeploy's install is taken from: PyPI (LMDeploy's wheel and the packages it needs; on Linux "
           "NVIDIA's CUDA 12 runtime, cuDNN and NCCL), PyTorch's own cu128 index (torch and torchvision built for CUDA "
           "12.8, and the rest of PyTorch's family by uv's --torch-backend) and Astral's mirror of the Python builds "
           "(uv's CPython; GitHub's releases after a failure). The plugin fetches LMDeploy's, torch's and "
           "torchvision's wheels itself, against their pinned sha256; the rest uv reads."),
    base=("pypi.org, files.pythonhosted.org; download.pytorch.org/whl/cu128 (its files on download-r2.pytorch.org); "
          "releases.astral.sh"),
    version=f"lmdeploy {release.VERSION}, torch {release.TORCH}+cu128, torchvision {release.TORCHVISION}+cu128",
    auth="none: read anonymously", limits="", costs="none",
    links=(
        Link("lmdeploy 0.18.0 on PyPI", "https://pypi.org/project/lmdeploy/0.18.0/", "tool"),
        Link("PyPI's JSON API", "https://docs.pypi.org/api/json/", "reference"),
        Link("PyTorch: Get Started", "https://pytorch.org/get-started/locally/", "guide"),
        Link("PyTorch's cu128 index", "https://download.pytorch.org/whl/cu128/", "docs"),
        Link("python-build-standalone", "https://github.com/astral-sh/python-build-standalone", "source"),
    ),
    areas=(
        Area("PyPI", "Python's package index and its file host", "https://docs.pypi.org/api/json/",
             covered="LMDeploy's wheel, checked against its pinned sum; what it needs, as PyPI gives it"),
        Area("PyTorch's cu128 index", "PyTorch's builds for CUDA 12.8", "https://pytorch.org/get-started/locally/",
             covered="torch and torchvision, checked against their pinned sums; the rest of its family, taken by uv"),
        Area("Python builds", "the CPython builds uv downloads", "https://github.com/astral-sh/python-build-standalone",
             covered="CPython 3.12.16, checked by uv against the sum built into it"),
    ),
    calls=(
        Call("GET LMDeploy's wheel", "LMDeploy 0.18.0's cp312 wheel for this system from PyPI's file host", "PyPI",
             used="the install's second download, checked against its pinned size and sha256",
             docs="https://pypi.org/project/lmdeploy/0.18.0/"),
        Call("GET torch's and torchvision's wheels", "their cp312 cu128 wheels for this system",
             "PyTorch's cu128 index",
             used="the install's third and fourth downloads, checked against their pinned sha256 (no size is listed)",
             docs="https://download.pytorch.org/whl/cu128/"),
        Call("GET /simple/{project}/", "an index's page of a project's files and their sums (what uv asks)", "PyPI",
             docs="https://docs.pypi.org/api/json/"),
    ),
    verified=VERIFIED)

NVIDIA_SMI = Api(
    name=NVIDIA, maker="NVIDIA", kind="command",
    about=("nvidia-smi, NVIDIA's System Management Interface, which comes with its driver. The plugin asks it for the "
           "cards' names, memory and driver, windowless, while the Models tab is shown and once inside an install "
           "(CUDA 12.8 needs driver 572.61 or newer on Windows, 570.124.06 on Linux), never for a status; under WSL it "
           "is Windows' driver's own, at /usr/lib/wsl/lib/nvidia-smi."),
    base="nvidia-smi (on PATH; under WSL /usr/lib/wsl/lib/nvidia-smi)",
    version="NVIDIA's driver's own",
    auth="none: what any user may read", limits="", costs="none",
    links=(
        Link("nvidia-smi", "https://docs.nvidia.com/deploy/nvidia-smi/index.html", "reference"),
        Link("CUDA 12.8 Update 1's release notes (the drivers it needs)",
             "https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html", "docs"),
        Link("CUDA on WSL", "https://docs.nvidia.com/cuda/wsl-user-guide/index.html", "guide"),
    ),
    areas=(
        Area("Queries", "--query-gpu: a card's properties as comma-separated values",
             "https://docs.nvidia.com/deploy/nvidia-smi/index.html",
             covered="index, name, memory (total, used, free) and driver of each card"),
        Area("Settings", "power limits, clocks and modes: changes, an administrator's",
             "https://docs.nvidia.com/deploy/nvidia-smi/index.html"),
    ),
    calls=(
        Call("nvidia-smi --query-gpu", "=index,name,memory.total,memory.used,memory.free,driver_version "
                                       "--format=csv,noheader,nounits", "Queries",
             used="the GPU table, what fits, and the install's driver check",
             docs="https://docs.nvidia.com/deploy/nvidia-smi/index.html"),
        Call("nvidia-smi -pl", "a card's power limit set", "Settings",
             docs="https://docs.nvidia.com/deploy/nvidia-smi/index.html", writes=True),
    ),
    verified=VERIFIED)

START_AT_SIGN_IN = Api(
    name=BOOT, maker="Microsoft (Task Scheduler) and the systemd project", kind="system",
    about=("An instance run here, started by the system when you sign in, on your confirmed click (At boot…): on "
           "Windows a scheduled task at your logon (your rights, never the highest), on Linux a systemd user unit. "
           "Each runs the platform's launcher (python -m liminal_libs.machines.launch serve), which reads the "
           "instance through your login at each start."),
    base="a task LiminalAI-lmdeploy-<instance>; a unit ~/.config/systemd/user/liminal-ai-lmdeploy-<instance>.service",
    version="Windows 11's ScheduledTasks module; systemd's user manager",
    auth="as you: no administrator, no root",
    limits="systemd as the init on Linux (a WSL distribution runs it only when /etc/wsl.conf says so)", costs="",
    links=(
        Link("Register-ScheduledTask", TASKS + "register-scheduledtask" + WS, "reference"),
        Link("Task Scheduler", LEARN + "windows/win32/taskschd/task-scheduler-start-page", "docs"),
        Link("systemd.service(5)", SYSTEMD + "systemd.service.html", "reference"),
        Link("systemctl(1)", SYSTEMD + "systemctl.html", "reference"),
    ),
    areas=(
        Area("Windows: a task at logon", "registered, read, started, stopped and removed",
             TASKS + "register-scheduledtask" + WS, covered="At boot… on Windows"),
        Area("Linux: a user unit", "written, enabled, started, stopped, disabled", SYSTEMD + "systemctl.html",
             covered="At boot… on Linux (systemd only: said where it does not run)"),
    ),
    calls=(
        Call("Register-ScheduledTask -AtLogOn", "a task at your logon, through the launcher",
             "Windows: a task at logon", used="At boot… on", writes=True, docs=TASKS + "register-scheduledtask" + WS),
        Call("Get-ScheduledTask | Get-ScheduledTaskInfo", "its state and last result", "Windows: a task at logon",
             used="the instance's state at boot", docs=TASKS + "get-scheduledtask" + WS),
        Call("Start-ScheduledTask / Stop-ScheduledTask", "the task run or ended now", "Windows: a task at logon",
             used="Start and Stop of an instance that starts at boot", writes=True,
             docs=TASKS + "start-scheduledtask" + WS),
        Call("Unregister-ScheduledTask", "the task removed", "Windows: a task at logon",
             used="At boot… off, Forget…", writes=True, docs=TASKS + "unregister-scheduledtask" + WS),
        Call("systemctl --user daemon-reload / enable", "the unit read and enabled for default.target",
             "Linux: a user unit", used="At boot… on", writes=True, docs=SYSTEMD + "systemctl.html"),
        Call("systemctl --user show", "the unit's state", "Linux: a user unit", used="the instance's state at boot",
             docs=SYSTEMD + "systemctl.html"),
        Call("systemctl --user start / stop", "the unit started or stopped now", "Linux: a user unit",
             used="Start and Stop of an instance that starts at boot", writes=True, docs=SYSTEMD + "systemctl.html"),
        Call("systemctl --user disable --now", "the unit stopped and disabled", "Linux: a user unit",
             used="At boot… off, Forget…", writes=True, docs=SYSTEMD + "systemctl.html"),
    ),
    verified=VERIFIED)

APIS = (LMDEPLOY_SERVER, LMDEPLOY_COMMAND, UV_TOOL, HF_COMMAND, HUB_API, PACKAGE_INDEXES, NVIDIA_SMI,
        START_AT_SIGN_IN)
