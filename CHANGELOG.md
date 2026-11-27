# Changelog

## 0.1.1 (2026-11-26)

Its lookup at the Hub and the lookups forgotten are asked only from the page too: Liminal's complete review
(Monday-list item 14, findings P2-1 and P3-1) and the user's word of 2026-11-26 for reads made with a key or rights,
"Page-only".

- **`find` is page-only** (`PAGE_ONLY_HOOKS`, no longer in `READ_HOOKS`). It asks huggingface.co about a repository
  named by the asker (`/api/models/<repo>` and its `/treesize/main`), with your Hugging Face token where one is kept.
  Until now any plugin, and any program on this machine through the platform, could make it ask about a model of its
  choosing under your token, with no click.
- **`clear_found` is page-only** too: it empties the lookups the Models tab keeps, a change and a choice of the person
  at the page, and was declared a read.
- Another plugin's ask and an ask that names no one are refused (403) in words before anything is sent, and
  liminal-platform asks its login whether you are at the page (401 otherwise). The Models tab asks both as the page,
  as before. With them every hook of its own says who may ask it.
- The model server's reads and its two services (`serving`, `models_of`) stay as they were.

Tests: a lookup and the lookups forgotten asked by another plugin and by no one are refused, the Hub is not asked and
what is kept stays; the page's lookup is made with the token as a bearer, and its clear_found empties them. They fail
when the two are put back in `READ_HOOKS`. No new requirement.

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
