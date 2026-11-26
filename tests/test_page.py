"""What the page draws: Main asks and runs nothing (no http, no command, no nvidia-smi); its own tables ("What leaves
this machine", "On this system") on Windows, on Linux, inside WSL with and without systemd, without nvidia-smi and
through a login that keeps nothing across a restart; the Models tab's Find a model, Downloaded here and, for an
instance elsewhere, what is not done from here; every form's fields its hook's arguments in order and its words naming
real fields; a button that does not apply disabled with its note; no token in any answer; the command line, which
reads alone and starts nothing."""
from __future__ import annotations

import inspect
import io
import json
import re
from contextlib import redirect_stderr, redirect_stdout

from liminal_libs.apps.authorization import EnvironmentLogin
from liminal_libs.machines.modelserver import ACCEPT_PLAIN
from plugin_platform_lmdeploy import __main__ as cli
from plugin_platform_lmdeploy import models, release

from conftest import TOKEN, data, make_install, make_model

REPO = "Qwen/Qwen3-0.6B"


def table(page: dict, title: str) -> dict:
    return next(t for t in page["tables"] if t["title"] == title)


def rows(page: dict, title: str) -> dict:
    return {r[0]: r for r in table(page, title)["rows"]}


def ready(world, system="linux", **more):
    p = world.plugin(system, machine="AMD64" if system == "windows" else "x86_64", **more)
    make_install(p)
    make_model(p, REPO)
    return p


# -- Main --------------------------------------------------------------------------------------------------------------
def test_main_asks_nothing_and_runs_nothing(world):
    p = ready(world)
    p.add_instance("default", "local", None, None, False, REPO)
    p.add_instance("gpu-box", "remote", "gpu-box:23333", None, True)
    p.start("default")
    out = p.status()
    assert out["valid"] is True and out["running"] == "1 of 1"
    assert world.transport.asked == [] and world.streamer.ran == [] and world.ran == []
    assert {t["title"] for t in out["tables"]} >= {"Instances", "What leaves this machine", "On this system"}
    assert out["pinned"] == "LMDeploy 0.18.0 (TurboMind; PyTorch 2.11.0 for CUDA 12.8), uv 0.13.8, CPython 3.12.16"


def test_on_windows_its_driver_turbomind_and_a_scheduled_task_are_said(world):
    out = ready(world, "windows").status()
    here = rows(out, "On this system")
    assert set(here) == {"GPU", "Driver", "Engine", "TurboMind on Windows", "Start at boot"}
    assert here["Driver"][1] == "572.61 or newer" and here["Engine"][1] == "TurboMind only"
    assert "PyTorch engine is never used here" in here["Engine"][2] and "one GPU" == here["TurboMind on Windows"][1]
    assert here["Start at boot"][1] == "a scheduled task" and "pinned" in out


def test_on_windows_an_lmdeploy_inside_wsl_is_offered_as_found_through_the_relay(world):
    world.probe.listen(5120, 23333, "wslrelay.exe")
    out = world.plugin("windows", machine="AMD64").status()
    answering = table(out, "Answering on this machine")
    (row,) = answering["rows"]
    assert "WSL's relay (wslrelay.exe, pid 5120)" in row[1]
    assert row[2]["form"] == "ai-add" and row[2]["values"] == {"name": "wsl", "where": "found", "port": 23333}
    assert world.transport.asked == [] and world.ran == []


def test_inside_wsl_its_gpu_its_memory_and_its_start_at_boot_are_said(world):
    here = rows(world.plugin().status(), "On this system")
    assert here["GPU"][1] == "nvidia-smi at /usr/bin/nvidia-smi" and "WSL 2: Ubuntu" in here["This Linux"][1]
    assert here["Driver"][1] == "570.124.06 or newer" and "Windows' driver, under WSL" in here["Driver"][2]
    assert "half of the PC's memory" in here["Memory (WSL)"][2] and here["Start at boot"][1] == "a systemd user unit"
    assert "TurboMind on Windows" not in here


def test_without_nvidia_smi_or_systemd_or_a_lasting_login_it_says_so(world):
    world.found.pop("nvidia-smi")
    p = world.plugin(systemd=False)
    here = rows(p.status(), "On this system")
    assert here["GPU"][1] == "nvidia-smi is not on this system" and "TurboMind needs an NVIDIA GPU" in here["GPU"][2]
    assert here["Start at boot"][1] == "not here" and "/etc/wsl.conf" in here["Start at boot"][2]
    lasting = world.plugin(login=False)
    lasting.login = EnvironmentLogin(env={})
    here = rows(lasting.status(), "On this system")
    assert here["Start at boot"][1] == "refused for now" and "until the platform restarts" in here["Start at boot"][2]


def test_a_linux_on_its_own_says_no_wsl_rows(world):
    here = rows(world.plugin(wsl="").status(), "On this system")
    assert "This Linux" not in here and "Memory (WSL)" not in here


def test_a_machine_it_does_not_run_on_says_so_and_pins_nothing(world):
    out = world.plugin("windows", machine="ARM64").status()
    assert "x86-64 alone" in rows(out, "On this system")["LMDeploy"][2] and "pinned" not in out


def test_its_ports_are_said_on_main(world):
    """The ports a tool opens by itself are said on its page (the user's answer 8): LMDeploy's one port, on 127.0.0.1
    alone, and no other (TurboMind on one GPU is one process)."""
    leaves = rows(world.plugin().status(), "What leaves this machine")
    assert leaves["Its port"][1] == "127.0.0.1 alone, the instance's own" and "opens no other port" in \
        leaves["Its port"][2]


# -- the Models tab ----------------------------------------------------------------------------------------------------
def test_with_no_instance_the_models_tab_offers_find_and_what_is_downloaded(world):
    p = ready(world)
    page = p.models()
    titles = [t["title"] for t in page["tables"]]
    assert titles[-2:] == ["Find a model", "Downloaded here"]
    assert {f["id"] for f in page["forms"]} == {"model-find", "model-pull", "model-delete"}
    find = table(page, "Find a model")
    assert find["search"] == {"form": "model-find", "field": "model", "value": "", "placeholder": models.PLACEHOLDER}
    (row,) = table(page, "Downloaded here")["rows"]
    assert row[0]["text"] == REPO and row[2] == "plain" and row[4] == "–"
    assert row[5]["form"] == "ai-add" and row[5]["values"] == {"where": "local", "model": REPO}
    assert row[6]["form"] == "model-delete" and not row[6].get("disabled")


def test_serve_it_changes_a_local_instance_shown_and_waits_while_it_runs(world):
    p = ready(world)
    make_model(p, "Org/Other-1B")
    p.add_instance("default", "local", None, 24000, False, "Org/Other-1B", "other", "awq", 8192, 50, "qwen3")
    p.choose("default")
    (row,) = [r for r in table(p.models(), "Downloaded here")["rows"] if r[0]["text"] == REPO]
    cell = row[5]
    assert cell["form"] == "ai-change" and cell["values"]["model"] == REPO and cell["values"]["port"] == 24000
    assert cell["values"]["share"] == 50 and cell["values"]["tools"] == "qwen3"
    assert all(cell["values"][k] is None for k in models.FOLLOW)  # the model's own settings follow the new one
    p.start("default")
    (row,) = [r for r in table(p.models(), "Downloaded here")["rows"] if r[0]["text"] == REPO]
    assert row[5]["disabled"] and row[5]["note"] == "stop default first: its model is read when it starts"


def test_a_download_is_offered_only_once_lmdeploy_is_installed(world):
    p = world.plugin()
    world.transport.answer("GET", f"/api/models/{REPO}", 200, data("hub-model.json"))
    p.find(REPO)
    cell = table(p.models(), "Find a model")["rows"][0][-1]
    assert cell["disabled"] and cell["note"].startswith("install LMDeploy first")


def test_an_instance_elsewhere_says_what_is_not_done_from_here(world):
    p = world.plugin("windows", machine="AMD64")
    p.add_instance("wsl", "found", None, 23333)
    world.transport.answer("GET", "/v1/models", 200, data("lmdeploy-models.json"))
    world.transport.answer("GET", "/health", 200, data("lmdeploy-health.json"))
    page = p.models()
    assert [t["title"] for t in page["tables"]][-1] == "wsl runs elsewhere"
    assert table(page, "wsl runs elsewhere")["note"] == models.ELSEWHERE
    assert rows(page, "wsl runs elsewhere")["Its model"][1] == "qwen3-0.6b"


# -- the forms ---------------------------------------------------------------------------------------------------------
def fields_of(form: dict) -> list[str]:
    return [f["name"] for f in form["fields"]]


def test_every_form_is_its_hooks_arguments_in_order_and_its_words_name_real_fields(world):
    p = ready(world)
    p.add_instance("default", "local", None, None, False, REPO)
    forms = [*p.status()["forms"], *p.models()["forms"]]
    assert {"model-find", "model-pull", "model-delete", "ai-add", "ai-change", "ai-start"} <= {f["id"] for f in forms}
    for form in forms:
        hook = getattr(p, form["hook"])
        params = inspect.signature(hook).parameters
        wanted = [a for a in params if a not in ("log",)]
        given = fields_of(form)
        if any(x.kind is inspect.Parameter.VAR_POSITIONAL for x in params.values()):
            head = [a for a in wanted if params[a].kind is not inspect.Parameter.VAR_POSITIONAL]
            assert given[:len(head)] == head, form["id"]
        else:
            assert given == wanted[:len(given)], form["id"]
        words = json.dumps(form.get("confirm") or {}) + " ".join(str(f.get("words", "")) for f in form["fields"])
        for name in re.findall(r"\{(\w+)\}", words):
            assert name in given, (form["id"], name)


def test_the_add_and_change_forms_carry_lmdeploys_settings_by_name(world):
    p = ready(world)
    p.add_instance("default", "local")
    forms = {f["id"]: f for f in p.status()["forms"]}
    names = ["model", "served", "format", "context", "share", "tools"]
    assert fields_of(forms["ai-add"])[-6:] == names and fields_of(forms["ai-change"])[-6:] == names
    assert fields_of(forms["ai-add"])[:5] == ["name", "where", "address", "port", ACCEPT_PLAIN]
    tools = next(f for f in forms["ai-add"]["fields"] if f["name"] == "tools")
    assert tools["type"] == "choice" and {c["value"] for c in tools["choices"]} >= {"qwen3coder", "llama3"}


def test_key_and_share_are_disabled_for_a_local_server_with_why(world):
    p = ready(world)
    p.add_instance("default", "local", None, None, False, REPO)
    text = json.dumps(p.status())
    assert "LMDeploy's server checks no key" in text and "LMDeploy's server has no key" in text


# -- no secret anywhere ----------------------------------------------------------------------------------------------
def test_no_token_is_in_any_answer(world):
    p = ready(world)
    p.login.kept["LIMINAL_AI_LMDEPLOY_HF_TOKEN"] = TOKEN
    p.add_instance("default", "local", None, None, False, REPO)
    world.transport.answer("GET", f"/api/models/{REPO}", 200, data("hub-model.json"))
    answers = [p.status(), p.models(), p.find(REPO), p.serving(), p.api_facts(), p.start("default"),
               p.log_of("default")]
    assert TOKEN not in json.dumps(answers, default=str)


# -- the command line --------------------------------------------------------------------------------------------------
def run(argv, p) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv, plugin=p)
    return code, out.getvalue(), err.getvalue()


def test_the_command_line_reads_and_starts_nothing(world):
    p = ready(world)
    p.add_instance("default", "local", None, None, False, REPO)
    code, out, _ = run(["status"], p)
    assert code == 0 and json.loads(out)["valid"] is True
    code, out, _ = run(["serving"], p)
    assert code == 0 and json.loads(out)["instances"][0]["name"] == "default"
    code, out, _ = run(["launch", "default"], p)
    assert code == 0 and "serve" in out and "api_server" in out and "--server-name" in out and "127.0.0.1" in out
    assert "api-keys" not in out
    assert run(["install"], p)[0] == 2 and run(["launch", "nobody"], p)[0] == 1
    assert world.streamer.ran == [] and world.kids.made == []


def test_the_command_line_asks_an_instance_for_its_models(world):
    p = world.plugin("windows", machine="AMD64")
    p.add_instance("wsl", "found", None, 23333)
    world.transport.answer("GET", "/v1/models", 200, data("lmdeploy-models.json"))
    world.transport.answer("GET", "/health", 200, data("lmdeploy-health.json"))
    code, out, _ = run(["models", "wsl"], p)
    assert code == 0 and [m["id"] for m in json.loads(out)["models"]] == ["qwen3-0.6b"]
    assert release.VERSION == "0.18.0"
