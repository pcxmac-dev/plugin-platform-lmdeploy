"""What the plugin attaches to (plugin API 2.2): its eight cards are sound (problems() is empty); each card names
exactly the calls the plugin makes of it (the modules' CALLS); the cards hold nothing of one machine; every outside
host the package names is on a card or among release.py's pins, and no address is written outside apis.py and
release.py; its facts ask nothing; and, with LIMINAL_VENDOR_PAGES=1 alone, every page on the cards answers (the only
test that reaches out)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from liminal_libs.sources import apis as A
from liminal_libs.sources.apis import problems, urls

import plugin_platform_lmdeploy as lmdeploy
from plugin_platform_lmdeploy import apis as cards
from plugin_platform_lmdeploy import models, release, server

SRC = Path(cards.__file__).resolve().parent
ADDRESS = re.compile(r"[a-z][a-z0-9+.-]*://[A-Za-z0-9\[][^\s\"')]*")


def card(api):
    return next(a for a in cards.APIS if a.name == api.name)


def test_the_cards_are_sound_and_each_says_where_its_documentation_is():
    assert problems(cards.APIS) == []
    assert [a.name for a in cards.APIS] == ["LMDeploy's api_server", "lmdeploy, the command", "uv",
                                            "hf, Hugging Face's command", "Hugging Face Hub API",
                                            "The package indexes", "nvidia-smi", "Start at sign-in"]
    assert all(a.verified == cards.VERIFIED and a.docs.startswith("https://") for a in cards.APIS)
    assert not any(c.costs for a in cards.APIS for c in a.calls)  # nothing here costs money


def test_each_card_names_exactly_the_calls_the_plugin_makes_of_it():
    made: dict[str, set[str]] = {}
    for calls in (release.CALLS, server.CALLS, models.CALLS):
        for name, said in calls.items():
            made.setdefault(name, set()).update(said)
    by_name = {a.name: a for a in cards.APIS}
    assert set(made) <= set(by_name)
    for name, api in by_name.items():
        assert {c.name for c in api.used} == made.get(name, set()), name


def test_the_servers_card_says_what_answers_without_a_key_and_what_is_switched_off():
    api = card(cards.LMDEPLOY_SERVER)
    calls = {c.name: c for c in api.calls}
    assert calls["POST /update_weights"].writes and not calls["POST /update_weights"].used
    assert not calls["POST /v1/responses"].used and not calls["GET /metrics"].used
    areas = {a.name: a for a in api.areas}
    assert "--disable-metrics" in areas["Metrics"].covered
    assert "--disable-fastapi-docs" in areas["Its API pages"].covered
    assert "owned_by 'lmdeploy'" in api.about and "never writes a key on a command line" in api.auth


def test_the_cards_say_what_holds_on_any_machine():
    for api in cards.APIS:
        words = [api.about, api.base, api.version, api.auth, *(a.about for a in api.areas),
                 *(c.about for c in api.calls)]
        assert not any(w and re.search(r"595\.71|RTX 5090 Laptop|this PC", w) for w in words), api.name


def _addresses(path: Path) -> list[str]:
    return ADDRESS.findall(path.read_text(encoding="utf-8"))


def test_no_address_is_written_outside_apis_and_release():
    for path in SRC.glob("*.py"):
        if path.name in ("apis.py", "release.py"):
            continue
        found = [a for a in _addresses(path) if not a.startswith("http://127.0.0.1")]
        assert found == [], path.name


def test_every_host_the_package_names_is_on_a_card_or_pinned():
    on_cards = {urlsplit(u).hostname for u in urls(cards.APIS)} | {urlsplit(a.base).hostname for a in cards.APIS
                                                                    if a.base.startswith("https://")}
    pinned = {*release.HOSTS, *release.ASSET_HOSTS, urlsplit(release.HUB).hostname}
    for path in (SRC / "apis.py", SRC / "release.py"):
        for address in _addresses(path):
            host = urlsplit(address).hostname
            assert host in on_cards | pinned | {"127.0.0.1"}, (path.name, address)
    assert {"github.com", "files.pythonhosted.org", "download-r2.pytorch.org", "huggingface.co"} <= on_cards | pinned


def test_the_readme_says_as_unconfirmed_what_only_a_real_run_shows():
    text = (SRC.parent.parent / "README.md").read_text(encoding="utf-8")
    unconfirmed = text.split("## Not yet confirmed by a real run", 1)[1].split("\n## ", 1)[0]
    for said in ("TurboMind's native module", "torch", "--no-build", "--allowed-media-domains localhost",
                 "Fallback to pytorch engine", "--allow-origins"):
        assert said in unconfirmed, said


def test_no_line_of_the_readme_changelog_or_workflow_is_longer_than_120():
    root = SRC.parent.parent
    for path in (root / "README.md", root / "CHANGELOG.md", root / ".github" / "workflows" / "ci.yml"):
        long = [n for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if len(line) > 120]
        assert not long, f"{path.name}: lines {long}"


def test_its_facts_say_what_it_learned_and_ask_nothing(world):
    p = world.plugin()
    assert set(p.api_facts()) == {cards.UV}  # nothing read yet, nothing asked
    before = (len(world.transport.asked), len(world.streamer.ran), len(world.ran))
    facts = A.facts_of(p)
    assert (len(world.transport.asked), len(world.streamer.ran), len(world.ran)) == before
    assert facts[cards.UV] == {"Pinned": f"uv {release.UV}, {release.PYTHON_SAID}"}


@pytest.mark.skipif(os.environ.get("LIMINAL_VENDOR_PAGES") != "1", reason="reaches the makers' pages: "
                                                                          "LIMINAL_VENDOR_PAGES=1")
def test_every_address_on_its_cards_answers(monkeypatch):
    monkeypatch.undo()  # this test alone reaches the network (the guard refuses it to every other)
    assert A.dead_links(lmdeploy.MACHINE.apis) == []
