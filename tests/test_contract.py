"""The plugin conforms to the class contract (plugin API 2.2): liminal_libs.sources.conformance, read from the plugin
alone (nothing runs). A machine plugin of the platform, a model server (ModelServer) drawn under the AI heading in the
Inference group at order 513, its icon LMDeploy's own mark; its hooks the base's and four of its own, the download a
cancellable job; its servers checking no key and speaking chat and completions; its floor and its version its
pyproject's; what it shares with the rest of the app declared (the Internal API tab's words: READ_HOOKS, KEEPS)."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import liminal_libs
import plugin_platform_lmdeploy as module
from liminal_libs import internal
from liminal_libs.machines.ai import version_tuple
from liminal_libs.machines.modelserver import SERVING, ModelServer
from liminal_libs.sources.conformance import assert_module_conforms
from liminal_libs.sources.keys import KeyVar
from liminal_libs.ui import STATIC

ROOT = Path(__file__).resolve().parent.parent
OWN = ("find", "clear_found", "pull", "delete_model")


def pyproject() -> str:
    return (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_the_plugin_conforms_to_the_class_contract():
    found = assert_module_conforms(module)
    assert [p.name for p in found] == ["lmdeploy"] and all(p.api == "2.2" and not p.is_record for p in found)
    assert {p.SYSTEM for p in found} == {"platform"} and {p.KIND for p in found} == {"machine"}
    p = found[0]
    assert isinstance(p, ModelServer) and (p.SECTION, p.GROUP, p.order, p.title) == ("ai", "inference", 513, "LMDeploy")
    assert p.interface()["apis"] == [a.name for a in p.apis]
    assert (p.DEFAULT_PORT, p.NATIVE, p.KEYED, p.READY_S, p.MODEL_COLUMNS) == (23333, "lmdeploy", False, 900.0, ())
    assert p.SPEAKS == ("chat", "completions") and "responses" not in p.SPEAKS
    assert internal.problems(p) == []


def test_its_hooks_are_the_bases_and_four_of_its_own():
    p = module.MACHINE
    hooks = set(p.interface()["hooks"])
    assert hooks == {"present", "status", "refresh", "install", "uninstall", "choose", "add_instance",
                     "change_instance", "remove_instance", "set_key", "forget_key", "share", "start", "stop", "boot",
                     "log_of", "models", "try_it", SERVING, "models_of", *OWN}
    assert p.HOOKS == (*ModelServer.HOOKS, *OWN)
    assert p.JOB_HOOKS == ("install", "pull") and p.CANCEL_HOOKS == ("install", "pull")
    assert "pull" in p.START_HOOKS and getattr(type(p).pull, "held", False)  # no install while hf downloads
    assert [t.hook for t in p.TABS] == ["models"]


def test_what_it_shares_with_the_app_is_declared():
    """Its own reads beside its bases', none a job and none taking an accept_ argument; what it keeps through the
    login; each hook it defines says itself in the first sentence of its docstring."""
    p = module.MACHINE
    assert p.READ_HOOKS == ("find", "clear_found") and set(p.READ_HOOKS) <= set(p.HOOKS)
    assert not set(p.READ_HOOKS) & set(p.JOB_HOOKS)
    for hook in p.READ_HOOKS:
        assert not [a for a in inspect.signature(getattr(p, hook)).parameters if a.startswith("accept_")]
    assert "SERVICE_HOOKS" not in type(p).__dict__
    assert [k[0] for k in p.KEEPS] == ["LIMINAL_AI_LMDEPLOY_INSTANCES", "LIMINAL_AI_LMDEPLOY_<INSTANCE>_KEY"]
    assert all(len(k) == 2 and all(isinstance(x, str) and x for x in k) for k in p.KEEPS)
    assert any("KEYED False" in n for n in p.INTERNAL_NOTES)
    for hook in OWN:
        doc = inspect.getdoc(getattr(type(p), hook)) or ""
        assert doc and doc.split(". ")[0].strip(), hook


def test_its_one_key_is_the_hugging_face_token():
    p = module.MACHINE
    assert p.keys == (KeyVar("LIMINAL_AI_LMDEPLOY_HF_TOKEN", label="Hugging Face token", hint=module.TOKEN.hint),)
    assert p.keys[0].secret and p.keys[0].hint.startswith("optional")
    assert p.variable("hf_token") == p.keys[0].name


def test_its_icon_is_its_makers_mark_drawn_by_the_library():
    drawn = set(re.findall(r"^\s*([a-z]+):\s*'", (STATIC / "icons.js").read_text(encoding="utf-8"), re.M))
    assert module.MACHINE.ICON == "lmdeploy" and "lmdeploy" in drawn


def test_its_version_and_floor_are_its_pyprojects():
    text = pyproject()
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == module.__version__ == "0.1.0"
    floor = re.search(r'"liminal-libs>=([0-9.]+)"', text).group(1)
    assert floor == module.MACHINE.LIBS_FLOOR == "1.66.0"
    assert version_tuple(liminal_libs.__version__) >= version_tuple(floor)
    assert 'lmdeploy = "plugin_platform_lmdeploy"' in text  # its entry point under liminal.providers


def test_a_copy_the_registry_hands_out_shares_what_the_plugin_keeps():
    p = module.LmdeployPlugin()
    copy = p.with_(origin="test")
    assert copy is not p and copy._rt is p._rt and copy.here is p.here


def test_making_it_runs_nothing(nothing_outside):
    module.LmdeployPlugin()
    assert nothing_outside == []


def test_no_line_of_the_package_is_longer_than_120():
    for path in [*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").glob("*.py"), ROOT / "pyproject.toml"]:
        long = [i for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if len(line) > 120]
        assert long == [], (path.name, long)
