"""The LMDeploy plugin from the command line, its reads only (the platform's page drives the same hooks):

    python -m plugin_platform_lmdeploy status          its Main, as JSON (files and kept readings: nothing run)
    python -m plugin_platform_lmdeploy serving         what it tells the other AI plugins: its instances (no key)
    python -m plugin_platform_lmdeploy models NAME     an instance asked now for its models (over http, a few seconds)
    python -m plugin_platform_lmdeploy launch NAME     a local instance's command as lines to paste (no key in it)

Instances are read through the login the plugin finds by itself (outside the platform: the environment's, your saved
variables on Windows). Nothing is installed, started, downloaded or changed from here. The exit code is 0 when the
answer is good, 1 when it is not, 2 for a command it does not know."""
from __future__ import annotations

import json
import sys

from liminal_libs.machines import children

from . import MACHINE

USAGE = "usage: python -m plugin_platform_lmdeploy status | serving | models NAME | launch NAME"


def _print(answer) -> None:
    print(json.dumps(answer, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None, plugin=None) -> int:
    words = list(sys.argv[1:] if argv is None else argv)
    p = plugin if plugin is not None else MACHINE
    command = words[0] if words else "status"
    try:
        if command == "status" and len(words) <= 1:
            _print(p.status())
            return 0
        if command == "serving" and len(words) <= 1:
            _print(p.serving())
            return 0
        if command == "models" and len(words) == 2:
            answer = p.models_of(words[1])
            _print(answer)
            return 0 if answer.get("ok") else 1
        if command == "launch" and len(words) == 2:
            instance = p.instance(words[1])
            why = p.startable(instance)
            if why:
                print(why, file=sys.stderr)
                return 1
            for line in children.shown(p.prepared(instance), p.system):
                print(line)
            return 0
    except (ValueError, LookupError) as e:
        print(" ".join(str(e).split()), file=sys.stderr)
        return 1
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
