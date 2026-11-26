"""What Install brings and what is installed (release.py), through the base's hooks as the page asks them: what is
offered on each system and processor and why not; what an install of this plugin's, a found one and none look like
(files looked at only); the install itself over stand-ins on Windows and on Linux (the four pinned files fetched and
checked, uv's archive unpacked, uv's two commands and the import check run in order with their whole environments,
the install put in use, its downloads let go); every refusal and failure in its words, the half-built folder removed;
a cancel; the uninstall, which stays inside the plugin's folder."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from liminal_libs import instances
from liminal_libs.machines.ai import ACCEPT_LICENCE, ACCEPT_REMOVAL
from liminal_libs.machines.world import size_words
from plugin_platform_lmdeploy import release

from conftest import Panel, install_effects, make_install, venv_bin

GITHUB, PYPI, TORCH = "https:" + "//github.com/", "https:" + "//files.pythonhosted.org/", "https:" + "//download-r2."
SYSTEMS = ("windows", "linux")


def test_the_pins_are_the_figures_read_from_the_makers_release_metadata(world):
    """The files this plugin fetches itself, by name, size, sha256 and address, as C-L1 and plugin-platform-aider's
    pins of uv read them: every other test stands their sums in with its own files', so a figure mistyped here shows
    nowhere else before the user's first install. (`world` stands the pins in; the module's own are read here from a
    fresh import of its source.)"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("pinned", release.__file__)
    pinned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pinned)
    assert (pinned.VERSION, pinned.TORCH, pinned.TORCHVISION, pinned.TORCH_BACKEND) == ("0.18.0", "2.11.0", "0.26.0",
                                                                                        "cu128")
    assert pinned.WHEELS["windows"][:3] == ("lmdeploy-0.18.0-cp312-cp312-win_amd64.whl", 58_831_511,
                                            "26c974dcedf547d0e868998002bda691575932ae77c65e0cb9b027946610603e")
    assert pinned.WHEELS["linux"][:3] == ("lmdeploy-0.18.0-cp312-cp312-manylinux_2_28_x86_64.whl", 106_692_457,
                                          "fcef7cf0d210d766e02e2e6df025378d2f8ae36080400b2818eb9a34ef3392dc")
    assert pinned.WHEELS["windows"][3] == (PYPI + "packages/77/d0/"
                                           "e2642c425be45809bb2059e77ae7aefdb503292cd9bd04d73b510552d320/")
    assert pinned.WHEELS["linux"][3] == (PYPI + "packages/19/8d/"
                                         "b880e02991ce03947c1efa9504c540d409b06d5d7ea4f95b1d8f8ae85c99/")
    assert pinned.TORCH_WHEELS["windows"] == (
        ("torch-2.11.0+cu128-cp312-cp312-win_amd64.whl",
         "7c78215c3af4f62e63f2b2e360f1722fc719b0853c7ac22666483d9810613a4c"),
        ("torchvision-0.26.0+cu128-cp312-cp312-win_amd64.whl",
         "8c0d1c4fbb2c9a4d5d41d0aaa87da20e525bcb2a154ce405725b0be59456804b"))
    assert pinned.TORCH_WHEELS["linux"] == (
        ("torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl",
         "d252cf975fb18c94a85336323ad425f473df56dab35a44b00399bd70c7a3b997"),
        ("torchvision-0.26.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl",
         "ccf26b4b659cfce6f2208cb8326071d51c70219a34856dfdf468d1e19af52c0d"))
    assert pinned.TORCH_FILES == TORCH + "pytorch.org/whl/cu128/"
    assert pinned.UV_FILES == {
        "windows": ("uv-x86_64-pc-windows-msvc.zip", 15_529_884,
                    "529e78a9cec6cd7e3e014f7ecd8bda0c113969d4ed0f2efeedb85c30943325b8"),
        "linux": ("uv-x86_64-unknown-linux-gnu.tar.gz", 16_855_069,
                  "5acfd9177cfecf0bf87578f059728215ccc884388988f55f7a65041b57387d91")}
    assert pinned.UV_RELEASE == GITHUB + "astral-sh/uv/releases/download/0.13.8/"
    assert pinned.DRIVERS == {"windows": "572.61", "linux": "570.124.06"}
    assert set(pinned.HOSTS) == {"github.com", "release-assets.githubusercontent.com", "releases.astral.sh", "pypi.org",
                                 "files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org"}
    assert pinned.NEED_FREE == 20 << 30


@pytest.mark.parametrize("system", SYSTEMS)
def test_the_sources_are_https_with_their_sums(world, system):
    p = world.plugin(system, machine="AMD64" if system == "windows" else "x86_64")
    uv = release.uv_source(p)
    assert uv.url.startswith(GITHUB) and uv.hosts == release.ASSET_HOSTS and uv.size and len(uv.sha256) == 64
    wheel = release.wheel_source(p)
    assert wheel.url.startswith(PYPI) and wheel.name.endswith(".whl") and wheel.size
    torch, vision = release.torch_sources(p)
    assert torch.url.startswith(TORCH) and "%2Bcu128" in torch.url and torch.name.startswith("torch-2.11.0+cu128")
    assert (torch.size, vision.size) == (0, 0) and len(torch.sha256) == len(vision.sha256) == 64  # no size is listed
    assert vision.name.startswith("torchvision-0.26.0+cu128")


def refusal(call) -> instances.Refusal:
    with pytest.raises(instances.Refusal) as caught:
        call()
    return caught.value


# -- what is offered where ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("system, machine", [("windows", "AMD64"), ("linux", "x86_64")])
def test_windows_x64_and_linux_x86_64_are_offered_the_pinned_install(world, system, machine):
    p = world.plugin(system, machine=machine)
    assert p.runs_here() == ""
    pkg = p.package()
    assert (pkg.title, pkg.version, pkg.pin, pkg.size) == ("LMDeploy", "0.18.0", "0.18.0", release.ABOUT[system])
    assert pkg.hosts == release.HOSTS and "download-r2.pytorch.org" in pkg.hosts
    assert pkg.licence.startswith("Apache-2.0, LMDeploy's own") and "NVIDIA's own proprietary terms" in pkg.licence
    assert "Ray (Apache-2.0)" in pkg.licence and "Each model carries its own licence" in pkg.licence
    assert "The size said is an estimate" in pkg.parts and "CPython 3.12.16" in pkg.parts
    assert "20.0 GB free is asked for" in pkg.parts and "nothing is built" in pkg.parts
    assert ("NVIDIA's CUDA 12 runtime" in pkg.parts) == (system == "linux")
    assert pkg.checked == release.CHECKED and "no sum pinned here checks it" in pkg.checked
    assert str(p.home / "models") in pkg.keeps
    form = next(f for f in p.status()["forms"] if f["id"] == "ai-install")
    assert form["job"] is True and release.CHECKED in form["confirm"]["words"]
    assert form["fields"][0]["name"] == ACCEPT_LICENCE


@pytest.mark.parametrize("system, given, words", [
    ("windows", {"machine": "ARM64"}, "x86-64 alone, and this machine is arm64"),
    ("linux", {"machine": "aarch64"}, "x86-64 alone, and this machine is aarch64"),
    ("linux", {"release_": "4.4.0-19041-Microsoft"}, "WSL 1"),
    ("linux", {"libc": ("musl", "1.2.5")}, "musl"),
    ("linux", {"libc": ("glibc", "2.27")}, "glibc 2.28 or newer (manylinux_2_28), and this Linux has glibc 2.27"),
])
def test_a_system_lmdeploy_cannot_run_on_is_told_in_words(world, system, given, words):
    p = world.plugin(system, **given)
    assert words in p.runs_here()
    assert words in refusal(lambda: p.install(True)).args[0]
    if "machine" in given:
        assert p.package() is None


def test_alpine_is_told_by_its_file(world):
    world.files.add("/etc/alpine-release")
    assert world.plugin().runs_here() == release.MUSL


def test_a_c_library_python_cannot_name_is_no_reason_to_refuse(world):
    assert world.plugin(libc=("", "")).runs_here() == ""


# -- what is installed -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("system", SYSTEMS)
def test_nothing_installed_and_no_lmdeploy_on_path(world, system):
    assert world.plugin(system, machine="AMD64").installed() is None


@pytest.mark.parametrize("system, end", [("windows", ".exe"), ("linux", "")])
def test_this_plugins_install_is_the_folder_current_names(world, system, end):
    p = world.plugin(system, machine="AMD64")
    folder = make_install(p)
    have = p.installed()
    assert (have.version, have.how, Path(have.path)) == ("0.18.0", "plugin", venv_bin(p, folder) / f"lmdeploy{end}")
    assert release.hf_of(p, have) == venv_bin(p, folder) / f"hf{end}"
    out = p.status()
    assert out["installed"].startswith("0.18.0, installed by this plugin in")
    assert not [f for f in out["forms"] if f["id"] == "ai-install"]  # the pinned install is the one in use
    assert world.ran == [] and world.streamer.ran == []  # files looked at: nothing run


def test_current_naming_a_folder_without_lmdeploy_is_no_install(world):
    p = world.plugin()
    folder = make_install(p)
    (venv_bin(p, folder) / "lmdeploy").unlink()
    world.found["lmdeploy"] = "/home/someone/.venv/bin/lmdeploy"  # (not looked for while CURRENT names one)
    assert p.installed() is None


@pytest.mark.parametrize("system, layout", [("linux", ("bin", "lmdeploy", "lib/python3.12/site-packages")),
                                            ("windows", ("Scripts", "lmdeploy.exe", "Lib/site-packages"))])
def test_an_lmdeploy_of_the_users_own_is_found_and_its_version_read_from_its_files(world, tmp_path, system, layout):
    venv = tmp_path / "users-venv"
    folder, program, packages = layout
    (venv / folder).mkdir(parents=True)
    (venv / folder / program).write_text("", encoding="utf-8")
    (venv / packages / "lmdeploy-0.17.2.dist-info").mkdir(parents=True)
    world.found["lmdeploy"] = str(venv / folder / program)
    p = world.plugin(system, machine="AMD64")
    have = p.installed()
    assert (have.version, have.how, have.note) == ("0.17.2", "found", release.FOUND_NOTE)
    assert "your own install: it is used as it is" in refusal(lambda: p.install(True)).args[0]
    assert world.ran == [] and world.streamer.ran == []


# -- the install -------------------------------------------------------------------------------------------------------
def install(world, p):
    install_effects(p, world.streamer)
    panel = Panel()
    return p.install(accept_licence=True, log=panel), panel


def test_the_install_is_asked_twice_its_licence_first(world):
    p = world.plugin()
    said = refusal(lambda: p.install())
    assert said.accept == ACCEPT_LICENCE and world.opener.asked == [] and world.ran == []


@pytest.mark.parametrize("system, machine", [("windows", "AMD64"), ("linux", "x86_64")])
def test_the_install_fetches_checks_and_runs_its_steps_in_order(world, system, machine):
    p = world.plugin(system, machine=machine)
    answer, panel = install(world, p)
    assert answer["ok"] is True, answer
    assert answer["said"].startswith("LMDeploy 0.18.0 installed in")
    uv_name, wheel = release.UV_FILES[system][0], release.WHEELS[system]
    torch, vision = release.TORCH_WHEELS[system]
    assert [u for u, _ in world.opener.asked] == [release.UV_RELEASE + uv_name, wheel[3] + wheel[0],
                                                  release.TORCH_FILES + torch[0].replace("+", "%2B"),
                                                  release.TORCH_FILES + vision[0].replace("+", "%2B")]
    assert world.streamer.keys() == ["uv venv", "uv pip", "python -I"]
    assert world.ran and world.ran[0][0].endswith("nvidia-smi")  # the driver read once, inside the job
    folder = p.home / "app" / "0.18.0"
    assert (p.home / "app" / "CURRENT").read_text(encoding="utf-8").strip() == "0.18.0"
    assert release.uv_in(p, folder).is_file() and p.installed().how == "plugin"
    venv = world.streamer.run_of("uv venv")
    assert venv["argv"][1:] == ["venv", "--python", "3.12", "--managed-python", "--no-project", "--no-config",
                                str(folder / "venv")]
    pip = world.streamer.run_of("uv pip")["argv"]
    downloads = p.home / "downloads"
    assert pip[-5:] == ["--no-build", "--torch-backend=cu128", str(downloads / wheel[0]), str(downloads / torch[0]),
                        str(downloads / vision[0])]
    assert "--extra-index-url" not in pip and "--index" not in pip  # PyTorch's index by its backend, never an extra
    check = world.streamer.run_of("python -I")
    assert check["argv"][0] == str(release.program_of(p, folder / "venv", "python"))
    assert check["argv"][1:3] == ["-I", "-c"] and "from lmdeploy.turbomind import _turbomind" in check["argv"][3]
    assert check["timeout"] == release.CHECK_S
    assert not os.listdir(downloads)  # every download let go once the install is whole
    assert any("checked against the sha256 pinned in this plugin" in line for line in panel.lines)
    assert release.CHECKED in answer["lines"]
    assert any("its TurboMind loading" in line for line in answer["lines"])


def test_every_command_of_the_install_runs_with_nothing_of_the_users_or_liminals(world):
    p = world.plugin()
    install(world, p)
    for run in world.streamer.ran:
        env = run["env"]
        assert "MASSIVE_API_KEY" not in env and not [k for k in env if k.startswith("LIMINAL_")], run["key"]
        assert env.get("UV_INDEX_URL") is None and env.get("PIP_INDEX_URL") is None and "PYTHONPATH" not in env
        assert "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env
        assert not [k for k in env if k.startswith(("LMDEPLOY_", "MODELSCOPE_", "TORCH_", "TRITON_"))], run["key"]
        assert env["UV_NO_CONFIG"] == "1" and env["UV_CACHE_DIR"] == str(p.home / "downloads" / "uv")
        assert env["UV_PYTHON_INSTALL_DIR"] == str(p.home / "app" / "0.18.0" / "uv" / "python")
        assert env["DO_NOT_TRACK"] == "1" and env["RAY_USAGE_STATS_ENABLED"] == "0"
        assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"


@pytest.mark.parametrize("system, gpu, words", [
    ("linux", None, "TurboMind needs an NVIDIA GPU: nvidia-smi reads none here"),
    ("linux", "0, NVIDIA GeForce RTX 4090, 24564, 1000, 23564, 570.86.10\n",
     "need NVIDIA's driver 570.124.06 or newer here; this one is 570.86.10 (it is Windows' driver: update it there)"),
    ("windows", "0, NVIDIA GeForce RTX 4090, 24564, 1000, 23564, 566.36\n",
     "need NVIDIA's driver 572.61 or newer here; this one is 566.36"),
])
def test_no_card_or_an_old_driver_ends_the_install_in_words(world, system, gpu, words):
    world.gpu = gpu
    p = world.plugin(system, machine="AMD64")
    answer, _ = install(world, p)
    assert answer["ok"] is False and words in answer["said"]
    assert world.opener.asked == [] and world.streamer.ran == [] and p.installed() is None


def test_a_driver_just_new_enough_is_taken(world):
    world.gpu = "0, NVIDIA GeForce RTX 4090, 24564, 1000, 23564, 572.61\n"
    p = world.plugin("windows", machine="AMD64")
    answer, _ = install(world, p)
    assert answer["ok"] is True, answer
    assert release.driver_tuple("570.124.06") > release.driver_tuple("570.86.10") and release.driver_tuple("") == ()


def test_too_little_room_ends_the_install_in_words(world):
    world.free = 5 << 30
    p = world.plugin()
    answer, _ = install(world, p)
    assert answer["ok"] is False and f"needs about {size_words(release.NEED_FREE)} free" in answer["said"]
    assert world.opener.asked == [] and world.streamer.ran == []


@pytest.mark.parametrize("which", ["UV_FILES", "TORCH_WHEELS"])
def test_a_wrong_sum_is_refused_and_nothing_is_put_in_use(world, monkeypatch, which):
    pins = dict(getattr(release, which))
    if which == "UV_FILES":
        name, size, _ = pins["linux"]
        pins["linux"] = (name, size, "0" * 64)
    else:
        (torch, _), vision = pins["linux"]
        pins["linux"] = ((torch, "0" * 64), vision)
    monkeypatch.setattr(release, which, pins)
    p = world.plugin()
    answer, _ = install(world, p)
    assert answer["ok"] is False and "not installed" in answer["said"]
    assert "uv pip" not in world.streamer.keys() and not (p.home / "app" / "CURRENT").exists()
    assert not (p.home / "app" / "0.18.0").exists()  # the half-built folder removed


def test_a_command_that_fails_ends_it_and_removes_the_half_built_folder(world):
    world.streamer.codes["uv pip"] = 2
    world.streamer.lines["uv pip"] = ["error: no solution found when resolving dependencies"]
    p = world.plugin()
    answer, panel = install(world, p)
    assert answer["ok"] is False and "`uv pip install` ended with 2" in answer["said"]
    assert "no solution found" in answer["said"] and "error: no solution found" in " ".join(panel.lines)
    assert not (p.home / "app" / "0.18.0").exists() and p.installed() is None
    assert (p.home / "downloads" / release.WHEELS["linux"][0]).is_file()  # kept for the next try


def test_turbomind_that_does_not_load_puts_nothing_in_use(world):
    """The import check is the install's last word: an environment whose TurboMind does not load would hand every
    model to the PyTorch engine (LMDeploy's command falls back silently), so it is never put in use."""
    world.streamer.codes["python -I"] = 1
    world.streamer.lines["python -I"] = ["ImportError: DLL load failed while importing _turbomind"]
    p = world.plugin("windows", machine="AMD64")
    answer, panel = install(world, p)
    assert answer["ok"] is False and "does not load LMDeploy 0.18.0's TurboMind" in answer["said"]
    assert "would fall back to its PyTorch engine" in answer["said"] and "DLL load failed" in " ".join(panel.lines)
    assert not (p.home / "app" / "CURRENT").exists() and p.installed() is None


def test_the_install_says_which_sum_covers_what_of_the_wheels(world):
    p = world.plugin()
    answer, panel = install(world, p)
    assert answer["ok"] is True, answer
    at = next(i for i, line in enumerate(panel.lines) if line.startswith("uv: ") and "into" in line
              and release.WHEELS["linux"][0] in line)
    assert "no sum pinned here checks it" in panel.lines[at]  # (the library's words of what it is given)
    ours = panel.lines[at - 1]
    assert ours.startswith(f"{release.WHEELS['linux'][0]}, torch-2.11.0+cu128")
    assert "uv's line below" in ours and "PyPI and PyTorch's index" in ours


def test_a_cancel_keeps_what_was_downloaded_and_installs_nothing(world):
    world.streamer.cancel_at = "python -I"
    p = world.plugin()
    answer, _ = install(world, p)
    assert answer["ok"] is False and answer["said"].startswith("cancelled: nothing was installed")
    assert not (p.home / "app" / "0.18.0").exists()
    assert (p.home / "downloads" / release.UV_FILES["linux"][0]).is_file()


def test_lmdeploy_not_found_after_uvs_install_puts_nothing_in_use(world):
    p = world.plugin()
    install_effects(p, world.streamer)
    world.streamer.effects.pop("uv pip")
    answer = p.install(accept_licence=True, log=Panel())
    assert answer["ok"] is False and "is not found in" in answer["said"]
    assert not (p.home / "app" / "CURRENT").exists() and "python -I" not in world.streamer.keys()


def test_the_time_given_each_step_holds_its_bytes_at_2_mbit():
    for seconds, size in ((release.PACKAGES_S, release.PACKAGES_BYTES), (release.TORCH_S, release.TORCH_BYTES),
                          (release.WHEEL_S, 106_692_457)):
        assert seconds >= size / release.RATE and seconds % 600 == 0
    assert release.seconds_for(1) == 600.0


# -- the uninstall -----------------------------------------------------------------------------------------------------
def test_the_uninstall_is_asked_twice_and_keeps_the_models(world):
    p = world.plugin()
    make_install(p)
    (p.home / "cache" / "torch").mkdir(parents=True)
    (p.home / "cache" / "torch" / "x").write_bytes(b"x" * 10)
    (p.home / "models" / "hub").mkdir(parents=True)
    (p.home / "downloads").mkdir()
    (p.home / "downloads" / "left.whl").write_bytes(b"left")
    outside = world.tmp / "beside-the-plugin"
    outside.mkdir()
    assert refusal(lambda: p.uninstall()).accept == ACCEPT_REMOVAL
    answer = p.uninstall(accept_removal=True)
    assert answer["ok"] is True, answer
    assert not (p.home / "app").exists() and not (p.home / "downloads").exists() and not (p.home / "cache").exists()
    assert (p.home / "models" / "hub").is_dir() and outside.is_dir()
    assert any("kept: its models" in line for line in answer["lines"])


def test_the_uninstall_says_before_it_is_asked_that_its_caches_go_with_it(world):
    p = world.plugin()
    make_install(p)
    form = next(f for f in p.status()["forms"] if f["id"] == "ai-uninstall")
    words = form["confirm"]["words"]
    assert f"Its caches ({p.home / 'cache'}) are deleted with it" in words
    assert f"its models ({p.home / 'models'}), its logs and your instances stay" in words


def test_a_found_install_is_never_removed_from_here(world, tmp_path):
    world.found["lmdeploy"] = str(tmp_path / "bin" / "lmdeploy")
    p = world.plugin()
    assert "your own install: it is never removed from here" in refusal(lambda: p.uninstall(True)).args[0]
    assert os.path.exists(tmp_path) and world.streamer.ran == []
