"""What the plugin needs of a machine (NEEDS: the needs plan's rows, written now in its getattr idiom). The library has
no needs module yet, so another class's NEEDS is read only through getattr (read by ast: a plain ModelServer.NEEDS
would fail to load on a library without it); each row is a part (an instance elsewhere serves anywhere), named once,
with its words. Once liminal_libs.needs exists, its own check of the rows is empty."""
from __future__ import annotations

import ast
import inspect

import pytest

import plugin_platform_lmdeploy as module

KINDS = {"part", "required"}


def test_another_classs_needs_are_read_through_getattr_alone():
    tree = ast.parse(inspect.getsource(module))
    plain = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "NEEDS"]
    assert plain == []  # no ModelServer.NEEDS anywhere
    read = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "getattr"
            and len(n.args) == 3 and isinstance(n.args[1], ast.Constant) and n.args[1].value == "NEEDS"]
    assert len(read) == 1 and isinstance(read[0].args[2], ast.Tuple) and not read[0].args[2].elts


def test_its_rows_are_parts_each_named_once_with_its_words():
    rows = module.MACHINE.NEEDS
    names = [r[0] for r in rows]
    assert names == ["architecture", "gpu", "gpu.driver", "libc", "wsl", "disk", "internet"]
    assert all(len(r) == 4 and all(isinstance(x, str) and x for x in r) and r[2] in KINDS for r in rows)
    assert {r[2] for r in rows} == {"part"}
    said = dict((r[0], r[1]) for r in rows)
    assert said["gpu.driver"] == ">=572.61 (Windows), >=570.124.06 (Linux)" and said["libc"] == "glibc>=2.28"
    assert said["wsl"] == "none|2" and said["disk"] == ">=20 GiB" and said["architecture"] == "x86-64"


def test_the_librarys_own_check_of_them_once_it_has_one():
    needs = pytest.importorskip("liminal_libs.needs", reason="the library has no needs module yet")
    assert needs.problems(module.MACHINE) == []
