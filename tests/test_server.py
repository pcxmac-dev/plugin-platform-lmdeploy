"""An LMDeploy server (server.py), through the base's hooks: its settings checked as they are typed; its command line
and its environment for each setting on both systems, always 127.0.0.1, TurboMind on one GPU, origins narrowed to its
own address, no media host, its docs and metrics off, no prompt logged, the executor without Ray, and no key on any
command line (it is never given one); what is taken out of the platform's environment; every refusal, a model
TurboMind does not serve among them (LMDeploy would fall back to its PyTorch engine) and GPT-OSS without its
vocabulary here; a start through the base; and what an instance answers (`look`): LMDeploy serving its model,
something that is not LMDeploy, a key wanted, an unhealthy or initializing engine, a local server that runs and does
not answer yet, and one whose log says it fell back to PyTorch, which is stopped. Nothing is started: the children
are stand-ins."""
from __future__ import annotations

import hashlib
import json

import pytest

from liminal_libs import instances, mounts
from liminal_libs.machines import children
from liminal_libs.machines.endpoint import Endpoint
from liminal_libs.machines.modelserver import Instance
from plugin_platform_lmdeploy import release, server

from conftest import SNAPSHOT, data, make_install, make_model, venv_bin

REPO = "Qwen/Qwen3-0.6B"
NEVER = ("--api-keys", "--proxy-url", "--allow-terminate-by-client", "--enable-abort-handling", "--ssl",
         "--trust-remote-code", "pytorch", "0.0.0.0", "*")


def refusal(call) -> instances.Refusal:
    with pytest.raises(instances.Refusal) as caught:
        call()
    return caught.value


def setting(name: str):
    return next(s for s in server.SETTINGS if s.name == name)


def ready(world, system="linux", **more):
    """A plugin with LMDeploy installed and Qwen3-0.6B downloaded."""
    p = world.plugin(system, machine="AMD64" if system == "windows" else "x86_64", **more)
    make_install(p)
    make_model(p, REPO)
    return p


def local(p, name="default", port=None, **settings):
    p.add_instance(name, "local", None, port, False, settings.pop("model", REPO), settings.pop("served", None),
                   settings.pop("format", None), settings.pop("context", None), settings.pop("share", None),
                   settings.pop("tools", None))
    return p.instance(name)


def snapshot(p, repo=REPO):
    return p.home / "models" / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / SNAPSHOT


# -- its settings ----------------------------------------------------------------------------------------------------
def test_the_settings_are_lmdeploys_main_ones_each_named_for_its_field():
    assert [s.name for s in server.SETTINGS] == ["model", "served", "format", "context", "share", "tools"]
    assert "backend" not in [s.name for s in server.SETTINGS]  # TurboMind alone: no engine to choose
    assert setting("share").default == 70 and (setting("share").min, setting("share").max) == (10, 95)
    assert (setting("context").min, setting("context").max) == (1024, 1_048_576)
    assert setting("format").default is None and setting("format").choices == server.FORMATS
    assert [c[0] for c in setting("tools").choices] == [c[0] for c in server.PARSERS]
    assert {"qwen3coder", "qwen3", "llama3", "internlm", "glm47", "interns2-preview"} <= {c[0] for c in server.PARSERS}


@pytest.mark.parametrize("name, typed, kept", [
    ("model", " Qwen/Qwen3-0.6B ", "Qwen/Qwen3-0.6B"), ("model", "/srv/models/qwen", "/srv/models/qwen"),
    ("model", "D:\\models\\qwen", "D:\\models\\qwen"), ("served", "qwen3-small", "qwen3-small"),
    ("tools", "qwen3coder", "qwen3coder"), ("format", "awq", "awq"), ("model", "", None)])
def test_a_setting_keeps_what_it_may_be(name, typed, kept):
    assert setting(name).clean(typed) == kept


@pytest.mark.parametrize("name, typed, words", [
    ("model", "qwen3", "org/name"), ("model", "Qwen/../etc", "org/name"), ("model", "a--b/c", "org/name"),
    ("served", "org/name", "no slash"), ("served", "two words", "no slash"), ("tools", "hermes", "one of qwen3coder"),
    ("format", "gguf", "format"), ("share", 99, "from 10 to 95"), ("context", 512, "from 1,024")])
def test_a_setting_refuses_what_it_may_not_be(name, typed, words):
    with pytest.raises(instances.Refusal) as caught:
        setting(name).clean(typed)
    assert words.lower() in str(caught.value).lower()


# -- the command line and the environment --------------------------------------------------------------------------
@pytest.mark.parametrize("system, end", [("windows", ".exe"), ("linux", "")])
def test_a_local_server_is_turbomind_offline_on_this_machine_alone(world, system, end):
    p = ready(world, system)
    launch = p.launch(local(p))
    program = venv_bin(p, p.home / "app" / "0.18.0") / f"lmdeploy{end}"
    assert launch.argv == (str(program), "serve", "api_server", str(snapshot(p)), "--server-name", "127.0.0.1",
                           "--server-port", "23333", "--model-name", "qwen3-0.6b", "--backend", "turbomind",
                           "--tp", "1", "--cache-max-entry-count", "0.70",
                           "--allow-origins", "http://127.0.0.1:23333", "--allowed-media-domains", "localhost",
                           "--disable-fastapi-docs", "--disable-metrics", "--log-level", "WARNING", "--max-log-len",
                           "0", "--distributed-executor-backend", "uni")
    for word in NEVER:
        assert word not in launch.argv, word
    env = launch.env
    assert {k: env[k] for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
                                "DO_NOT_TRACK")} == dict.fromkeys(("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                                                                   "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"), "1")
    assert env["LMDEPLOY_USE_MODELSCOPE"] == env["LMDEPLOY_USE_OPENMIND_HUB"] == "False"
    assert env["LMDEPLOY_EXECUTOR_BACKEND"] == "uni" and env["RAY_USAGE_STATS_ENABLED"] == "0"
    assert env["UVICORN_LOG_LEVEL"] == "warning" and env["HF_HOME"] == str(p.home / "models")
    assert env["TIKTOKEN_ENCODINGS_BASE"] == str(p.home / "tiktoken")
    for name in ("TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TIKTOKEN_RS_CACHE_DIR"):
        assert env[name].startswith(str(p.home / "cache")), name
    assert ("XDG_CACHE_HOME" in env) == (system == "linux")
    assert launch.cwd == str(p.home) and launch.secret == ()
    assert set(server.DROP) <= set(launch.drop)


def test_each_setting_becomes_its_switch(world):
    p = ready(world)
    launch = p.launch(local(p, port=24000, served="qwen-small", share=55, context=32768, tools="qwen3",
                            format="awq"))
    words = list(launch.argv)
    assert words[words.index("--model-name") + 1] == "qwen-small"
    assert words[words.index("--cache-max-entry-count") + 1] == "0.55"
    assert words[words.index("--session-len") + 1] == "32768"
    assert words[words.index("--tool-call-parser") + 1] == "qwen3"
    assert words[words.index("--model-format") + 1] == "awq"
    assert words[words.index("--server-port") + 1] == "24000"
    assert words[words.index("--allow-origins") + 1] == "http://127.0.0.1:24000"  # its own address, its own port


def test_no_key_is_ever_given_and_none_can_be_kept_for_a_local_server(world):
    """LMDeploy takes keys on its command line alone (--api-keys, C-L1 1f), where every program of this machine reads
    them, so a local server checks none (KEYED False): Key… and Share… are refused for it by the base, and the lines
    to paste carry no key."""
    p = ready(world)
    instance = local(p)
    assert "checks no key" in refusal(lambda: p.set_key("default", "sk-a-key-0123456789")).args[0]
    assert "has no key" in refusal(lambda: p.share("default", "yes", accept_share=True)).args[0]
    launch = p.prepared(instance)
    assert "--api-keys" not in launch.argv and launch.secret == ()
    shown = children.shown(launch, "linux", base={})
    assert not any("api-keys" in line for line in shown)
    assert "LIMINAL_*" in launch.drop and "MASSIVE_API_KEY" in launch.drop  # every pair of Liminal's (dropped())


@pytest.mark.parametrize("system", ["windows", "linux"])
def test_nothing_of_the_platforms_environment_reconfigures_a_server(world, system):
    p = ready(world, system)
    whole = children.environment(p.prepared(local(p)), base=p.out.env, system=system)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "TRANSFORMERS_CACHE", "PYTHONPATH", "MODELSCOPE_CACHE",
                 "RAY_ADDRESS", "SSL_KEYFILE", "LMDEPLOY_LOG_FILE", "MASSIVE_API_KEY", "LIMINAL_AI_LMDEPLOY_INSTANCES",
                 "LIMINAL_PLATFORM_DIR"):
        assert name not in whole, name
    assert whole["LMDEPLOY_USE_MODELSCOPE"] == "False" and whole["LMDEPLOY_EXECUTOR_BACKEND"] == "uni"
    assert whole["HF_HUB_OFFLINE"] == "1" and whole["UVICORN_LOG_LEVEL"] == "warning"
    assert whole["TORCH_HOME"] == str(p.home / "cache" / "torch")
    assert whole["TIKTOKEN_ENCODINGS_BASE"] == str(p.home / "tiktoken") and whole["PYTHONUNBUFFERED"] == "1"


def test_a_start_goes_through_the_base_as_a_windowless_child(world):
    p = ready(world)
    local(p)
    answer = p.start("default")
    assert answer["ok"], answer
    child = world.kids.made[-1]
    assert child.launch.argv[1:3] == ("serve", "api_server") and child.launch.env["HF_HUB_OFFLINE"] == "1"
    assert p.status()["running"] == "1 of 1"


def test_a_start_needs_lmdeploy_installed(world):
    p = world.plugin()
    make_model(p, REPO)
    local(p)
    assert "LMDeploy is not installed here: Install… first" in refusal(lambda: p.start("default")).args[0]
    assert "LMDeploy is not installed here" in refusal(lambda: p.launch(p.instance("default"))).args[0]
    assert world.kids.made == []


def test_a_start_refuses_a_model_not_set_not_downloaded_or_not_whole(world):
    p = world.plugin()
    make_install(p)
    p.add_instance("default", "local")
    assert "set default's model first (Change…)" in refusal(lambda: p.start("default")).args[0]
    local(p, "other", 23334, model="Org/Elsewhere-7B")
    assert "download Org/Elsewhere-7B first (Models)" in refusal(lambda: p.launch(p.instance("other"))).args[0]
    make_model(p, "Org/Half-1B", incomplete=True)
    local(p, "half", 23335, model="Org/Half-1B")
    assert "download is not whole" in refusal(lambda: p.launch(p.instance("half"))).args[0]


@pytest.mark.parametrize("config, words", [
    ({"architectures": ["MistralForCausalLM"]}, "MistralForCausalLM, which LMDeploy's TurboMind does not serve"),
    ({"architectures": ["Qwen3ForCausalLM"], "quantization_config": {"quant_method": "smooth_quant"}},
     "quantized by SmoothQuant"),
    ({}, "no config.json that can be read"),
])
def test_a_model_turbomind_does_not_serve_is_refused_before_lmdeploy_could_fall_back(world, config, words):
    """LMDeploy's command hands a model TurboMind does not serve to its PyTorch engine with a warning alone (C-L1
    1g), and that engine always holds an unauthenticated loopback port in a second process (4b): so such a model is
    refused here, in words, and nothing is started."""
    p = world.plugin()
    make_install(p)
    make_model(p, "Org/Other-7B", config=config)
    local(p, model="Org/Other-7B")
    said = refusal(lambda: p.start("default")).args[0]
    assert words in said and ("PyTorch engine, which is never used here" in said)
    assert world.kids.made == []


def test_every_architecture_turbomind_serves_is_started():
    assert len(server.TURBOMIND) == 19 and len(set(server.TURBOMIND)) == 19
    assert {"Qwen3ForCausalLM", "LlamaForCausalLM", "GptOssForCausalLM", "MixtralForCausalLM",
            "InternLM2ForCausalLM"} <= set(server.TURBOMIND)


def test_a_turbomind_workspace_is_turbominds_own(world, tmp_path):
    p = world.plugin()
    make_install(p)
    folder = tmp_path / "workspace"
    (folder / "triton_models").mkdir(parents=True)
    local(p, "ws", model=str(folder))
    launch = p.launch(p.instance("ws"))
    assert launch.argv[3] == str(folder) and launch.argv[launch.argv.index("--model-name") + 1] == "workspace"


def test_gpt_oss_needs_its_vocabulary_in_the_plugins_folder_whole(world, monkeypatch):
    """openai-harmony downloads o200k_base.tiktoken at a GPT-OSS server's first request unless a folder holds it
    (C-L1 5b): a server here reaches nothing by itself, so a GPT-OSS start is refused until that file is there and is
    the one openai-harmony checks."""
    p = world.plugin()
    make_install(p)
    make_model(p, "openai/gpt-oss-20b", config={"architectures": ["GptOssForCausalLM"]})
    local(p, model="openai/gpt-oss-20b")
    said = refusal(lambda: p.launch(p.instance("default"))).args[0]
    assert "o200k_base.tiktoken" in said and str(p.home / "tiktoken") in said
    vocab = p.home / "tiktoken" / server.TIKTOKEN
    vocab.parent.mkdir(parents=True)
    vocab.write_bytes(b"not the vocabulary")
    assert "the file there is not that one" in refusal(lambda: p.launch(p.instance("default"))).args[0]
    monkeypatch.setattr(server, "TIKTOKEN_SHA256", hashlib.sha256(b"not the vocabulary").hexdigest())
    assert p.launch(p.instance("default")).argv[1:3] == ("serve", "api_server")


def test_a_folder_model_is_its_whole_path_on_this_systems_own_disk(world, tmp_path, monkeypatch):
    p = world.plugin()
    make_install(p)
    folder = tmp_path / "my-model"
    local(p, "mine", model=str(folder))
    assert "is not a folder here" in refusal(lambda: p.launch(p.instance("mine"))).args[0]
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(data("qwen3-config.json")), encoding="utf-8")
    launch = p.launch(p.instance("mine"))
    assert launch.argv[3] == str(folder) and launch.argv[launch.argv.index("--model-name") + 1] == "my-model"
    monkeypatch.setattr(mounts, "foreign", lambda path, *a, **k: True)
    assert "another system's drive" in refusal(lambda: p.launch(p.instance("mine"))).args[0]


def test_what_was_kept_otherwise_is_refused_at_its_start(world):
    p = ready(world)
    bad = Instance("odd", settings={"model": REPO, "served": "org/name", "share": 70})
    assert "holds a slash or a space" in refusal(lambda: p.launch(bad)).args[0]
    wide = Instance("wide", settings={"model": REPO, "share": 99})
    assert "10 to 95" in refusal(lambda: p.launch(wide)).args[0]
    parser = Instance("parser", settings={"model": REPO, "tools": "hermes"})
    assert "none LMDeploy 0.18.0 registers" in refusal(lambda: p.launch(parser)).args[0]
    form = Instance("form", settings={"model": REPO, "format": "gguf"})
    assert "none LMDeploy names" in refusal(lambda: p.launch(form)).args[0]


def test_harnesses_are_told_the_name_a_server_answers_to_and_what_it_speaks(world):
    p = ready(world)
    local(p)
    local(p, "small", 23334, served="qwen-small")
    p.add_instance("box", "remote", "gpu-box:23333", None, True)
    rows = {r["name"]: r for r in p.serving()["instances"]}
    assert rows["default"]["model"] == "qwen3-0.6b" and rows["small"]["model"] == "qwen-small"
    assert rows["box"]["model"] == "" and p.serving("small")["instances"][0]["model"] == "qwen-small"
    assert all(r["speaks"] == ["chat", "completions"] for r in rows.values())  # no Responses API
    assert p.serving()["native"] == "lmdeploy" and world.transport.asked == [] and world.streamer.ran == []


# -- words -------------------------------------------------------------------------------------------------------------
def test_start_words_say_what_it_holds_where_it_listens_and_that_it_checks_no_key(world):
    p = ready(world)
    said = p.start_words(local(p, share=60))
    assert f"It loads {REPO} at once with LMDeploy's TurboMind, then keeps 60% of the GPU's free memory" in said
    assert "started offline" in said and "binds 127.0.0.1, although LMDeploy's own default is every interface" in said
    assert "no other web page can read it" in said and "opens its one port and no other" in said
    assert "checks no key" in said


def test_what_leaves_this_machine_and_its_ports_are_said(world):
    rows = {r[0]: r for r in server.leaves(world.plugin())["rows"]}
    assert set(rows) == {"Its port", "Web pages", "Hugging Face", "Images in a request", "GPT-OSS's vocabulary",
                         "Telemetry", "Its API pages and metrics", "Install"}
    assert "opens no other port" in rows["Its port"][2] and "/update_weights" in rows["Its port"][2]
    assert "ModelScope and openMind's hub are off" in rows["Hugging Face"][2]
    assert "a file of this machine's" in rows["Images in a request"][2]
    assert rows["Install"][1] == "on your click"


# -- what an instance answers ----------------------------------------------------------------------------------------
def answering(world, models=None, health=200, said=None):
    t = world.transport
    t.answer("GET", "/v1/models", 200, models if models is not None else data("lmdeploy-models.json"))
    t.answer("GET", "/health", health, said if said is not None else data("lmdeploy-health.json"))


def at(p, instance) -> Endpoint:
    return p.endpoint(instance)


def test_an_lmdeploy_answering_lists_its_model(world):
    p = ready(world)
    instance = local(p)
    answering(world)
    seen = p.look(at(p, instance), instance)
    assert seen.ok and seen.version == "0.18.0" and "it serves qwen3-0.6b" in seen.said
    (model,) = seen.models
    assert (model["id"], model["loaded"], model["tools"]) == ("qwen3-0.6b", True, False)
    assert model["detail"] == "qwen3-0.6b · 70% of the free GPU memory for its cache"
    assert seen.facts == {"engine": "TurboMind", "metrics": "off", "health": "healthy"}
    assert [a["path"] for a in world.transport.asked] == ["/v1/models", "/health"]


def test_a_found_lmdeploys_engine_is_not_known_from_here(world):
    p = world.plugin("windows", machine="AMD64")
    p.add_instance("wsl", "found", None, 23333)
    instance = p.instance("wsl")
    answering(world)
    seen = p.look(at(p, instance), instance)
    assert seen.ok and seen.facts["engine"].startswith("its own") and seen.version == ""
    assert seen.models[0]["tools"] is None


def test_what_answers_with_another_owner_is_not_lmdeploy(world):
    p = ready(world)
    instance = local(p)
    world.transport.answer("GET", "/v1/models", 200, {"object": "list", "data": [{"id": "x", "owned_by": "vllm"}]})
    seen = p.look(at(p, instance), instance)
    assert not seen.ok and "is not LMDeploy" in seen.said


def test_a_remote_lmdeploy_that_wants_a_key_says_so(world):
    p = ready(world)
    p.add_instance("box", "remote", "gpu-box:23333", None, True)
    instance = p.instance("box")
    world.transport.answer("GET", "/v1/models", 401, {"error": "Unauthorized"})
    seen = p.look(at(p, instance), instance)
    assert not seen.ok and "wants a key" in seen.said and "(Key…)" in seen.said


@pytest.mark.parametrize("health, said, words", [
    (503, {"status": "unhealthy", "message": "engine died"}, "it says it is unhealthy: Stop, then Start"),
    (200, {"status": "initializing", "message": ""}, "initializing, for qwen3-0.6b"),
])
def test_an_engine_that_is_not_healthy_says_so(world, health, said, words):
    p = ready(world)
    instance = local(p)
    answering(world, health=health, said=said)
    seen = p.look(at(p, instance), instance)
    assert not seen.ok and words in seen.said


def test_a_server_that_runs_and_does_not_answer_yet_is_said_so(world):
    p = ready(world)
    local(p)
    p.start("default")
    world.transport.silent.add(("GET", "/v1/models"))
    instance = p.instance("default")
    assert p.look(at(p, instance), instance).said == server.NOT_YET


def test_a_server_whose_log_says_it_fell_back_to_pytorch_is_stopped(world):
    """The second guard behind the refusal before a start: should LMDeploy all the same hand its model to PyTorch (its
    TurboMind module not loading, say), its log's warning is read when the instance is looked at, and the server is
    stopped, said so; nothing is asked of it."""
    p = ready(world)
    local(p)
    p.start("default")
    child = world.kids.made[-1]
    log = p.log_path("default")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("2026-11-26 01:00:00,000 - lmdeploy - WARNING - archs.py:51 - Fallback to pytorch engine because "
                   "turbomind is not built.\n", encoding="utf-8")
    instance = p.instance("default")
    seen = p.look(at(p, instance), instance)
    assert not seen.ok and "handed its model to its PyTorch engine" in seen.said and "it is stopped" in seen.said
    assert seen.facts["engine"].startswith("fell back") and child.stops == 1 and not child.alive
    assert world.transport.asked == []
