"""LMDeploy's models (models.py, on the library's hub.py): looked up at the Hub's API (found, gated, another licence,
unknown, no answer, kept an hour and 12 at most, cleared, the token sent only where one is kept); downloaded as a job
with the environment's hf on both systems (each plain refusal before its first line, the second confirmation built
from the lookup without an ask, hf's whole environment, its progress, a failure, a cancel that keeps the part, the
token in none of its lines, no thread left); the models folder read and kept, each model's architecture and format
read from its config.json; deleted (refused while served or downloading); and what fits the GPU."""
from __future__ import annotations

import json
import threading
import time
from urllib.parse import parse_qs, urlsplit

import pytest

from liminal_libs import instances, mounts
from liminal_libs.machines import gpu
from liminal_libs.machines.ai import ACCEPT_LICENCE, HIDDEN
from liminal_libs.machines.terminal import Cancelled
from plugin_platform_lmdeploy import models, release

from conftest import TOKEN, Panel, data, make_install, make_model, venv_bin

REPO = "Qwen/Qwen3-0.6B"
GATED = "meta-llama/Llama-3.2-1B"


def refusal(call) -> instances.Refusal:
    with pytest.raises(instances.Refusal) as caught:
        call()
    return caught.value


def hub(world):
    t = world.transport
    t.answer("GET", f"/api/models/{REPO}", 200, data("hub-model.json"))
    t.answer("GET", f"/api/models/{REPO}/treesize/main", 200, data("hub-treesize.json"))
    t.answer("GET", f"/api/models/{GATED}", 200, data("hub-gated.json"))
    t.answer("GET", f"/api/models/{GATED}/treesize/main", 401, {"error": "Access to model is restricted"})
    t.answer("GET", "/api/models/someone/Odd-Model-7B", 200, data("hub-other.json"))
    t.answer("GET", "/api/models/someone/Odd-Model-7B/treesize/main", 200, {"path": "/", "size": 14_000_000_000})


def keep_token(p) -> None:
    p.login.kept["LIMINAL_AI_LMDEPLOY_HF_TOKEN"] = TOKEN


# -- looking a model up ----------------------------------------------------------------------------------------------
def test_a_model_looked_up_says_its_size_files_and_licence_and_is_kept(world):
    p = world.plugin()
    hub(world)
    answer = p.find(REPO)
    assert answer["ok"] is True and answer["said"] == f"{REPO}: 1.4 GB to download in 10 files; licence: apache-2.0"
    assert answer["lines"][0].startswith("the weights (1.4 GB) fit in the NVIDIA GeForce RTX 5090 Laptop GPU's")
    assert "a server then keeps 70% of what is left" in answer["lines"][0]
    assert answer["lines"][1] == "read at huggingface.co" and "config.json" in answer["lines"][2]
    ask = world.transport.asked[0]
    assert urlsplit(ask["url"]).hostname == "huggingface.co" and "Authorization" not in ask["headers"]
    assert parse_qs(urlsplit(ask["url"]).query) == {"expand": ["cardData", "gated", "siblings"]}
    (found,) = models.lookups(p)
    assert (found.model, found.size, found.files, found.licence, found.gated) == (REPO, 1519209243, 10, "apache-2.0",
                                                                                  False)
    assert found.page == "https:" + "//huggingface.co/Qwen/Qwen3-0.6B/blob/main/LICENSE"
    assert p._rt.last == REPO and world.streamer.ran == []


def test_a_gated_model_says_it_and_its_download_waits_for_a_token(world):
    p = world.plugin()
    make_install(p)
    hub(world)
    answer = p.find(GATED)
    assert answer["ok"] is True and "not said to download in 13 files; licence: llama3.2" in answer["said"]
    assert any("gated: its files need the Hugging Face token" in line for line in answer["lines"])
    found = models.lookup(p, GATED)
    assert found.gated and found.size is None and "private or gated" in found.why
    assert "it needs the Hugging Face token" in refusal(lambda: p.pull(GATED)).args[0]
    cell = models._find_table(p, None, "")["rows"][0][-1]
    assert cell["disabled"] and "no Hugging Face token is kept" in cell["note"]


def test_a_token_kept_goes_to_the_hub_as_a_bearer_and_into_no_answer(world):
    p = world.plugin()
    keep_token(p)
    hub(world)
    world.transport.answer("GET", f"/api/models/{GATED}/treesize/main", 200, {"path": "/", "size": 2_471_645_608})
    answer = p.find(GATED)
    assert all(a["headers"].get("Authorization") == f"Bearer {TOKEN}" for a in world.transport.asked)
    assert TOKEN not in json.dumps(answer) and models.lookup(p, GATED).size == 2_471_645_608


def test_another_licence_is_said_by_its_name_and_its_page(world):
    p = world.plugin()
    hub(world)
    p.find("someone/Odd-Model-7B")
    found = models.lookup(p, "someone/Odd-Model-7B")
    assert found.licence == "odd-community-licence"
    assert found.page == "https:" + "//huggingface.co/someone/Odd-Model-7B/blob/main/LICENSE.md"


@pytest.mark.parametrize("status, words", [(404, "knows no model by that name"), (429, "fewer requests"),
                                           (401, "private or gated")])
def test_a_model_the_hub_does_not_say_is_kept_with_why(world, status, words):
    p = world.plugin()
    world.transport.answer("GET", "/api/models/Org/Nothing", status, {"error": "no"})
    answer = p.find("Org/Nothing")
    assert answer["ok"] is False and words in answer["said"]
    assert words in models.lookup(p, "Org/Nothing").why


def test_no_answer_from_the_hub_is_said(world):
    p = world.plugin()
    world.transport.silent.add(("GET", "/api/models/Org/Nothing"))
    answer = p.find("Org/Nothing")
    assert answer["ok"] is False and "did not say its size or licence" in answer["said"]


def test_a_name_that_is_none_is_refused_before_anything_is_asked(world):
    p = world.plugin()
    assert "org/name" in refusal(lambda: p.find("qwen3")).args[0]
    assert world.transport.asked == []


def test_lookups_are_kept_an_hour_twelve_at_most_newest_first(world):
    p = world.plugin()
    for n in range(14):
        world.transport.answer("GET", f"/api/models/Org/M{n}", 200, {"id": f"Org/M{n}", "siblings": []})
        p.find(f"Org/M{n}")
    kept = [f.model for f in models.lookups(p)]
    assert len(kept) == 12 and kept[0] == "Org/M13" and "Org/M0" not in kept
    world.clock.now += 3601
    assert models.lookups(p) == []


def test_clear_forgets_the_lookups(world):
    p = world.plugin()
    hub(world)
    p.find(REPO)
    assert p.clear_found()["ok"] and models.lookups(p) == [] and p._rt.last == ""


# -- downloading -------------------------------------------------------------------------------------------------------
def looked_up(world, system="linux", **more):
    p = world.plugin(system, machine="AMD64" if system == "windows" else "x86_64", **more)
    make_install(p)
    hub(world)
    p.find(REPO)
    return p


def test_a_download_is_refused_plainly_before_its_first_line(world):
    p = world.plugin()
    hub(world)
    assert "install LMDeploy first" in refusal(lambda: p.pull(REPO)).args[0]
    make_install(p)
    assert "look Org/Other-1B up first (Find a model)" in refusal(lambda: p.pull("Org/Other-1B")).args[0]
    p.find(REPO)
    world.free = 1 << 30
    assert "needs 1.4 GB" in refusal(lambda: p.pull(REPO)).args[0]
    world.free = 500 << 30
    p._rt.pulling = "Org/Busy"
    assert "one download at a time" in refusal(lambda: p.pull(REPO)).args[0]
    arm = world.plugin("windows", machine="ARM64")
    assert "x86-64 alone" in refusal(lambda: arm.pull(REPO)).args[0]
    assert world.streamer.ran == []


def test_its_second_confirmation_says_size_licence_and_room_from_the_lookup_alone(world):
    p = looked_up(world)
    asked, ran = len(world.transport.asked), len(world.ran)
    said = refusal(lambda: p.pull(REPO))
    assert said.accept == ACCEPT_LICENCE and len(world.transport.asked) == asked  # nothing asked again
    assert len(world.ran) == ran  # nothing run: it is said within the platform's one second
    words = said.args[0]
    assert words.startswith(f"{REPO}: 1.4 GB in 10 files from huggingface.co, into ")
    assert "Licence: apache-2.0; its words are at https:" in words and "Whether TurboMind serves it" in words
    assert world.streamer.ran == []


@pytest.mark.parametrize("system, end", [("windows", ".exe"), ("linux", "")])
def test_the_download_runs_the_environments_hf_with_its_whole_environment(world, system, end):
    p = looked_up(world, system)
    panel = Panel()

    def download(argv, env):
        assert p._rt.pulling == REPO and p._rt.pull_panel is panel  # a detach cancels it, as the page's Cancel does
        make_model(p, REPO, config={"architectures": ["Qwen3ForCausalLM"],
                                    "quantization_config": {"quant_method": "awq", "bits": 4}})

    world.streamer.effects["hf download"] = download
    answer = p.pull(REPO, accept_licence=True, log=panel)
    assert answer["ok"] is True and answer["said"].startswith(f"{REPO} is here (")
    assert "Qwen3ForCausalLM: TurboMind serves it" in answer["lines"]
    assert any(line.startswith("its weights are AWQ") for line in answer["lines"])
    run = world.streamer.run_of("hf download")
    assert run["argv"][0] == str(venv_bin(p, p.home / "app" / "0.18.0") / f"hf{end}")
    assert run["argv"][1:] == ["download", REPO, "--quiet"]
    env = run["env"]
    assert env["HF_HOME"] == str(p.home / "models") and env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    for name in ("HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_UPDATE_CHECK", "HF_HUB_DISABLE_SHARED_BLOBS",
                 "HF_HUB_DISABLE_PROGRESS_BARS", "DO_NOT_TRACK"):
        assert env[name] == "1", name
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "LMDEPLOY_USE_MODELSCOPE", "PYTHONPATH", "MASSIVE_API_KEY",
                 "LIMINAL_PLATFORM_DIR", "TRANSFORMERS_CACHE", "MODELSCOPE_CACHE", "TORCH_HOME", "RAY_ADDRESS"):
        assert name not in env, name
    assert run["timeout"] == models.PULL_S and panel.lines[0].startswith(f"{REPO} is downloaded from huggingface.co")
    assert run["cwd"] == str(models.hub(p))
    assert p._rt.pulling == "" and p._rt.pull_panel is None and models.cached(p, REPO) is not None


def test_a_kept_token_goes_to_hf_by_environment_and_out_of_its_lines(world):
    p = looked_up(world)
    keep_token(p)
    world.streamer.lines["hf download"] = [f"using the token {TOKEN} for this download"]
    world.streamer.effects["hf download"] = lambda argv, env: make_model(p, REPO)
    panel = Panel()
    answer = p.pull(REPO, accept_licence=True, log=panel)
    env = world.streamer.run_of("hf download")["env"]
    assert env["HF_TOKEN"] == TOKEN and "HF_HUB_DISABLE_IMPLICIT_TOKEN" not in env
    assert TOKEN not in " ".join(world.streamer.run_of("hf download")["argv"])
    assert TOKEN not in " ".join(panel.lines) and f"using the token {HIDDEN}" in " ".join(panel.lines)
    assert TOKEN not in json.dumps(answer)


def test_a_download_that_fails_says_so(world):
    p = looked_up(world)
    world.streamer.codes["hf download"] = 1
    answer = p.pull(REPO, accept_licence=True, log=Panel())
    assert answer["ok"] is False and "hf ended with 1" in answer["said"] and p._rt.pulling == ""


def test_hf_ending_well_without_the_model_whole_is_said(world):
    p = looked_up(world)
    world.streamer.effects["hf download"] = lambda argv, env: make_model(p, REPO, incomplete=True)
    answer = p.pull(REPO, accept_licence=True, log=Panel())
    assert answer["ok"] is False and "is not whole" in answer["said"]


def test_a_cancelled_download_keeps_what_came_and_says_where(world):
    p = looked_up(world)
    world.streamer.effects["hf download"] = lambda argv, env: make_model(p, REPO, incomplete=True)
    world.streamer.cancel_at = "hf download"
    answer = p.pull(REPO, accept_licence=True, log=Panel())
    assert answer["ok"] is False and answer["said"].startswith(f"cancelled: what came of {REPO} stays in")
    assert p._rt.pulling == "" and p._rt.pull_panel is None
    (kept,) = models.entries(p, fresh=True)
    assert kept.incomplete  # the part is kept, for Delete… or a download again


def test_its_progress_is_said_from_the_folders_bytes(world, monkeypatch):
    p = looked_up(world)
    monkeypatch.setattr(models, "EVERY_S", 0.01)
    panel = Panel()

    def download(argv, env):
        make_model(p, REPO)
        for _ in range(200):
            if any("of 1.4 GB" in line for line in panel.lines):
                return
            time.sleep(0.01)

    world.streamer.effects["hf download"] = download
    p.pull(REPO, accept_licence=True, log=panel)
    assert any(line.startswith(f"{REPO}: ") and "of 1.4 GB" in line for line in panel.lines)


def waiting_streamer(seen: dict, started: threading.Event):
    """An hf that downloads until it is cancelled (3 s at most): whether its cancel was seen, in `seen`."""
    def streamer(argv, line, *, env, cwd=None, timeout=None, cancel=None):
        started.set()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if cancel is not None and cancel():
                seen["cancelled"] = True
                raise Cancelled("hf download was cancelled: ended with what it started")
            time.sleep(0.01)
        seen["cancelled"] = False
        return 0
    return streamer


def test_turning_the_plugin_off_cancels_a_download(world):
    seen, started = {}, threading.Event()
    p = looked_up(world, streamer=waiting_streamer(seen, started))
    panel, out = Panel(), {}
    pull = threading.Thread(target=lambda: out.update(answer=p.pull(REPO, accept_licence=True, log=panel)),
                            name="test-pull", daemon=True)
    pull.start()
    try:
        assert started.wait(5)
        p.detach()
    finally:
        pull.join(10)
    assert seen["cancelled"] is True
    assert out["answer"]["said"].startswith(f"cancelled: what came of {REPO}") and p._rt.pull_panel is None


def test_install_and_uninstall_wait_for_a_download_and_say_why(world):
    p = world.plugin()
    make_install(p, "0.17.0")  # an older pin of this plugin's: Update… and Uninstall… are offered
    p._rt.pulling = REPO
    actions = {a["form"]: a for a in p.status()["actions"] if a.get("form")}
    for form in ("ai-install", "ai-uninstall"):
        assert actions[form]["disabled"] and actions[form]["note"].startswith(f"{REPO} is being downloaded"), form
    assert refusal(lambda: p.install(accept_licence=True)).args[0].startswith(f"{REPO} is being downloaded")
    assert refusal(lambda: p.uninstall(accept_removal=True)).args[0].startswith(f"{REPO} is being downloaded")
    p._rt.pulling = ""
    actions = {a["form"]: a for a in p.status()["actions"] if a.get("form")}
    assert not actions["ai-install"].get("disabled") and not actions["ai-uninstall"].get("disabled")
    assert world.streamer.ran == [] and world.opener.asked == []


def tables_of(page: dict) -> dict:
    return {t["title"]: t for t in page["tables"]}


def test_a_folder_on_another_systems_drive_takes_no_download_and_no_delete(world, monkeypatch):
    p = looked_up(world)
    make_model(p, REPO)
    monkeypatch.setattr(mounts, "foreign", lambda path, *a, **k: True)
    words = p.home_problem()
    assert words and refusal(lambda: p.pull(REPO, accept_licence=True)).args[0] == words
    assert refusal(lambda: p.delete_model(REPO)).args[0] == words
    page = tables_of(p.models())
    cell = page["Find a model"]["rows"][0][-1]
    assert cell["disabled"] and cell["note"] == words
    row = page["Downloaded here"]["rows"][0]
    assert row[5]["disabled"] and row[6]["disabled"] and row[6]["note"] == words
    assert world.streamer.ran == [] and models.cached(p, REPO) is not None


# -- the models folder -------------------------------------------------------------------------------------------------
def test_the_models_folder_is_read_from_its_files_and_kept_a_minute(world):
    p = world.plugin()
    make_model(p, REPO)
    (entry,) = models.entries(p)
    assert (entry.repo, entry.incomplete, entry.whole) == (REPO, False, True) and entry.size >= 4096
    assert entry.snapshot == "c1899de289a04d12100db370d81485cdf75e47ca"
    make_model(p, "Org/Later-1B", incomplete=True)
    assert [e.repo for e in models.entries(p)] == [REPO]  # the reading kept
    world.clock.now += 61
    later = {e.repo: e for e in models.entries(p)}
    assert set(later) == {REPO, "Org/Later-1B"} and later["Org/Later-1B"].incomplete


def test_what_a_downloaded_model_is_is_read_from_its_config(world):
    p = world.plugin()
    make_install(p)
    make_model(p, REPO)
    make_model(p, "mistralai/Mistral-7B-v0.3", config={"architectures": ["MistralForCausalLM"]})
    make_model(p, "Org/Fp8-8B", config={"architectures": ["LlamaForCausalLM"],
                                        "quantization_config": {"quant_method": "fp8"}})
    rows = {r[0]["text"]: r for r in tables_of(p.models())["Downloaded here"]["rows"]}
    assert rows[REPO][0]["sub"] == "Qwen3ForCausalLM: TurboMind serves it" and rows[REPO][2] == "plain"
    assert rows["Org/Fp8-8B"][2] == "FP8"
    mistral = rows["mistralai/Mistral-7B-v0.3"]
    assert "TurboMind does not serve it" in mistral[0]["sub"]
    assert mistral[5]["disabled"] and "PyTorch engine, which is never used here" in mistral[5]["note"]  # Serve it…
    assert not rows[REPO][5].get("disabled")


# -- deleting ----------------------------------------------------------------------------------------------------------
def test_a_model_is_deleted_with_its_lock_files(world):
    p = world.plugin()
    make_install(p)
    folder = make_model(p, REPO)
    locks = p.home / "models" / "hub" / ".locks" / folder.name
    locks.mkdir(parents=True)
    (locks / "x.lock").write_text("", encoding="utf-8")
    p.add_instance("idle", "local", None, None, False, REPO)
    answer = p.delete_model(REPO)
    assert answer["ok"] is True and answer["said"].startswith(f"{REPO} is deleted from")
    assert "freed" in answer["said"] and not folder.exists() and not locks.exists()
    assert any("idle is set to serve it" in line for line in answer["lines"])
    assert models.cached(p, REPO) is None


def test_a_model_not_here_downloading_or_served_is_not_deleted(world):
    p = world.plugin()
    make_install(p)
    assert "is not downloaded here" in refusal(lambda: p.delete_model(REPO)).args[0]
    make_model(p, REPO)
    p._rt.pulling = REPO
    assert "Cancel it in the panel first" in refusal(lambda: p.delete_model(REPO)).args[0]
    p._rt.pulling = ""
    p.add_instance("default", "local", None, None, False, REPO)
    p.start("default")
    assert "stop default first: it serves Qwen/Qwen3-0.6B" in refusal(lambda: p.delete_model(REPO)).args[0]
    assert models.cached(p, REPO) is not None


# -- what fits ---------------------------------------------------------------------------------------------------------
def card(total_mib: int, free_mib: int) -> gpu.Gpus:
    return gpu.Gpus((gpu.Gpu(0, "RTX", total_mib << 20, (total_mib - free_mib) << 20, free_mib << 20, "595.71"),))


@pytest.mark.parametrize("size, gpus, share, words", [
    (4 << 30, card(24576, 20480), 70, "the weights (4.0 GB) fit in the RTX's 20.0 GB free now; a server then keeps "
                                      "70% of what is left (11.2 GB) for its cache"),
    (22 << 30, card(24576, 20480), 70, "are larger than what the RTX has free now (20.0 GB): LMDeploy would not "
                                       "start with it"),
    (4 << 30, gpu.Gpus(why="nvidia-smi is not on this system"), 70, "no NVIDIA GPU is read here"),
    (None, card(24576, 20480), 70, "its size is not said"),
])
def test_what_fits_the_gpu_is_said(size, gpus, share, words):
    assert words in models.fit_words(size, gpus, share)


def test_the_hub_is_named_by_this_plugin_and_reached_alone(world):
    p = world.plugin()
    hub(world)
    p.find(REPO)
    assert world.transport.hosts() == {"huggingface.co"} and release.HUB.endswith("//huggingface.co")
