"""Who may ask the plugin's own changes (the rule of the sweep after the review's finding PLT-H1, 2026-11-25). The
platform hands an ask from any program on this PC to a plugin's hook, so the plugin says which of the hooks it adds
only its page asks (PAGE_ONLY_HOOKS, as liminal-libs 1.66.0's apps/asking.py and liminal-platform 0.34.0 read them): a
model downloaded into its models folder, and a downloaded model's files deleted. Another plugin's ask of one is
refused, and so is an ask that names no one: LMDeploy's hf is not run and no file is deleted. The page's ask does
both as before. The model server's services are answered to anyone. Its lookup (find) and the lookups forgotten
(clear_found) are the page's alone too (0.1.1; the user, 2026-11-26, of reads made with a key: "Page-only"): find asks
huggingface.co about a name of the asker's choosing with the Hugging Face token kept, and clear_found empties what the
Models tab keeps. It names its own alone: the model server's base says its own (liminal-libs 1.99.0), and the
names add up along the class line.

Asked through the library's own App (its ask names who asks as the platform's does), over this suite's stand-ins
(conftest.py): nothing is run or downloaded, and its guard fails a test that reaches out."""
from __future__ import annotations

import pytest

pytest.importorskip("liminal_libs.apps.asking", reason="no hook is asked only from its page before liminal-libs 1.66.0")

from liminal_libs import internal  # noqa: E402 - after the skip, as each of these
from liminal_libs.apps import App, asking  # noqa: E402
from liminal_libs.machines.modelserver import ModelServer  # noqa: E402
from liminal_libs.machines.plugin import MachinePlugin  # noqa: E402

from plugin_platform_lmdeploy import MACHINE, LmdeployPlugin, models  # noqa: E402
from conftest import Panel, make_install, make_model  # noqa: E402
from conftest import TOKEN  # noqa: E402
from test_models import GATED, REPO, hub, keep_token  # noqa: E402

HERE = "someone/Odd-Model-7B"  # a model downloaded already: the one to delete
CHANGES = ("pull", "delete_model")  # what it adds to the model server's changes
PAGE_READS = ("find", "clear_found")  # a lookup at the Hub, and the lookups kept forgotten: the page's since 0.1.1
BASES = tuple(getattr(ModelServer, "PAGE_ONLY_HOOKS", ()))  # the base's own (liminal-libs 1.99.0's; none before)
ASKS = {  # each change as an ask that would be done, a download's licence accepted: an argument the asker sets
    "pull": (REPO, True),
    "delete_model": (HERE,),
}


class Neighbour(MachinePlugin):
    """Another plugin of the platform, asking LMDeploy's plugin through its app as any plugin may (a harness, or the MCP
    plugin for a local agent)."""

    name, title = "neighbour", "Neighbour"

    def wants(self, hook, *args):
        """Not a hook: one of LMDeploy's plugin's asked through the app."""
        return self.app.ask("lmdeploy", hook, *args)


class Platform(App):
    """The library's App asking as the platform does: from its page's route, the page named in so many words, or from
    its own code, no one named."""

    NAME, TITLE, ICON = "platform", "Platform", "chip"

    def from_page(self, hook, *args, **more):
        """As the page's route asks a hook (a job's with its panel as `log`)."""
        with asking.asking(asking.PAGE):
            return self.ask("lmdeploy", hook, *args, **more)

    def on_its_own(self, hook, *args):
        """As the app's own code asks a hook."""
        return self.ask("lmdeploy", hook, *args)


@pytest.fixture
def attached(world):
    """The plugin on the library's App beside another plugin, with its install, a model looked up at the Hub and
    another downloaded already; let go after."""
    p = world.plugin(attach=False)
    app, neighbour = Platform(log=lambda m: None, login=p.login), Neighbour()
    app.attach(p)
    app.attach(neighbour)
    make_install(p)
    hub(world)
    p.find(REPO)
    folder = make_model(p, HERE)
    yield app, neighbour, p, folder
    app.detach(neighbour.name)
    app.detach(p.name)


def test_its_own_hooks_are_said_to_be_asked_only_from_its_page():
    assert LmdeployPlugin.PAGE_ONLY_HOOKS == (*CHANGES, *PAGE_READS) and set(ASKS) == set(CHANGES)  # its own alone
    assert set(internal.page_only(MACHINE)) == {*BASES, *CHANGES, *PAGE_READS}  # what the platform reads
    assert internal.plugin_only(MACHINE) == ()
    assert [p for p in internal.problems(MACHINE) if "_ONLY_HOOKS" in p] == []  # each a hook of its own
    assert set(MACHINE.HOOKS) >= {*CHANGES, *PAGE_READS}
    opened = {*getattr(MACHINE, "SERVICE_HOOKS", ()), *internal.declared(MACHINE, "READ_HOOKS")}
    assert not {*CHANGES, *PAGE_READS} & opened
    for hook in (*CHANGES, *PAGE_READS):
        assert asking.refusal(MACHINE, hook, asking.PAGE) == ""
        assert asking.refusal(MACHINE, hook, "neighbour") == (
            f"lmdeploy's {hook} is asked only from its page, never by another plugin: neighbour asked it.")
        assert asking.refusal(MACHINE, hook, None).endswith("This ask names no asker: neither the page nor a plugin.")
    for hook in ("status", "serving", "models_of"):  # as before, and what a harness asks of it
        assert {asking.refusal(MACHINE, hook, who) for who in (asking.PAGE, "neighbour", None)} == {""}


def test_another_plugin_or_no_one_asking_a_change_runs_and_deletes_nothing_and_its_page_does(attached, world):
    app, neighbour, p, folder = attached
    world.streamer.effects["hf download"] = lambda argv, env: make_model(p, REPO)
    ran, sent = len(world.streamer.ran), len(world.transport.asked)
    for hook in CHANGES:  # each as a program would send it, and each would be done
        for ask in (lambda: neighbour.wants(hook, *ASKS[hook]), lambda: app.on_its_own(hook, *ASKS[hook])):
            with pytest.raises(asking.AskRefused, match=f"lmdeploy's {hook} is asked only from its page") as refused:
                ask()
            assert refused.value.status == 403 and refused.value.hook == hook
    assert len(world.streamer.ran) == ran and len(world.transport.asked) == sent  # hf not run, the Hub not asked
    assert folder.exists() and models.cached(p, REPO) is None and p._rt.pulling == ""  # nothing deleted or downloaded
    # its page's asks are done, as before
    assert app.from_page("delete_model", *ASKS["delete_model"])["ok"] and not folder.exists()
    panel = Panel()
    assert app.from_page("pull", *ASKS["pull"], log=panel)["ok"] and models.cached(p, REPO) is not None
    assert world.streamer.run_of("hf download")["argv"][1:] == ["download", REPO, "--quiet"]
    assert isinstance(neighbour.wants("status"), dict) and "instances" in neighbour.wants("serving")


def test_another_plugin_or_no_one_asking_a_lookup_sends_nothing_to_the_hub_and_its_page_does(attached, world):
    """0.1.1 (the user, 2026-11-26, of reads made with a key: "Page-only"): a lookup another plugin or an ask naming no
    one words, of a model of its choosing, is refused before anything is sent, so huggingface.co is not asked and the
    Hugging Face token kept goes nowhere; the lookups kept are not forgotten for them either. The page's lookup is made,
    the token with it, and its clear_found empties them, as before."""
    app, neighbour, p, folder = attached
    keep_token(p)
    world.transport.answer("GET", f"/api/models/{GATED}/treesize/main", 200, {"path": "/", "size": 2_471_645_608})
    sent, kept = len(world.transport.asked), [f.model for f in models.lookups(p)]
    assert kept == [REPO]
    for hook, args in (("find", (GATED,)), ("clear_found", ())):
        for ask in (lambda: neighbour.wants(hook, *args), lambda: app.on_its_own(hook, *args)):
            with pytest.raises(asking.AskRefused, match=f"lmdeploy's {hook} is asked only from its page") as refused:
                ask()
            assert refused.value.status == 403 and refused.value.hook == hook
    assert len(world.transport.asked) == sent  # the Hub not asked, the token sent nowhere
    assert [f.model for f in models.lookups(p)] == kept and models.lookup(p, GATED) is None  # kept, nothing new
    assert app.from_page("find", GATED)["ok"] and models.lookup(p, GATED).size == 2_471_645_608
    made = world.transport.asked[sent:]
    assert made and all(a["headers"].get("Authorization") == f"Bearer {TOKEN}" for a in made)
    assert app.from_page("clear_found")["ok"] and models.lookups(p) == []

