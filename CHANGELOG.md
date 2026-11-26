# Changelog

## 0.1.0 (2026-11-26)

The first release: LMDeploy as the fourth of the platform's top four inference engines (the user, 2026-11-18: "top 4
inference engines, top 4 agentic harnesses (open source)", "both linux and windows compatibility"; 2026-11-25:
LMDeploy, TurboMind only, never the silent PyTorch fallback). A model server on liminal-libs 1.66.0's AI bases
(`ModelServer`, plugin API 2.2), `SECTION = "ai"`, `GROUP = "inference"`, `order = 513`, `ICON = "lmdeploy"`: the
standard interface of a model server, and four hooks of its own for its models. Needs liminal-libs>=1.99.0, whose
model server base asks its changes only from the page.

- **Install, Update, Uninstall** from the page, on Windows x64 and on Linux x86-64 with glibc 2.28 or newer (WSL 2
  included): uv 0.13.8, an environment on uv's CPython 3.12.16, LMDeploy 0.18.0's cp312 wheel and PyTorch 2.11.0's
  torch and torchvision wheels for CUDA 12.8, each fetched file checked against its pinned sha256, then
  `uv pip install --no-build --torch-backend=cu128`, then a windowless check that TurboMind's native module loads; put
  in use only then. The GPU, a driver older than CUDA 12.8 needs (572.61 on Windows, 570.124.06 on Linux) and too
  little room (20 GB) are refused first, in words. Its licence and what is not checked are said before it starts.
- **A server per model**, started offline as a windowless child (or at boot by a scheduled task or a systemd user
  unit): `lmdeploy serve api_server` on the snapshot's folder, on 127.0.0.1, TurboMind on one GPU, cross-origin
  answers narrowed to its own address, no media host, its docs and metrics off, no prompt logged, Ray kept out, its
  caches in the plugin's folder, the platform's own variables kept out. No key on any command line: its servers
  check none (`KEYED = False`), and none is shared beyond this machine from here.
- **TurboMind only**: a model whose config.json names an architecture TurboMind does not serve, a SmoothQuant model,
  or one without a config.json is refused before a start (LMDeploy's command would hand it to its PyTorch engine,
  which holds an unauthenticated loopback port in a second process); a server whose log says it fell back all the same
  is stopped when it is looked at. GPT-OSS is refused until its vocabulary is in the plugin's own folder.
- **Models** on the library's hub.py: looked up at the Hub's API (size, files, licence, gate), downloaded as a
  cancellable job with the environment's hf, listed with each one's architecture and format, deleted.
- **Its page** says what leaves this machine and its one port, what this system lacks, and its eight API cards. Its
  rows say it speaks chat and completions (no Responses API). Its `NEEDS` rows are written in the getattr idiom.
