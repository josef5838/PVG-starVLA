#!/usr/bin/env python3
"""Verify a SurfloDiT bundle and strictly load it with the bundled code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--surflo-root", required=True, type=Path)
    parser.add_argument("--check-files-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.bundle.resolve() / "code"))

    from starVLA.model.framework.surflo_bundle import resolve_surflo_bundle

    checkpoint, overrides, manifest = resolve_surflo_bundle(args.bundle, args.surflo_root)
    if not args.check_files_only:
        from starVLA.model.framework.base_framework import baseframework

        model = baseframework.from_pretrained(str(checkpoint), config_overrides=overrides)
        if model.config.framework.name != manifest["framework"]:
            raise ValueError("Loaded framework differs from bundle manifest")
        if int(model.config.framework.action_model.num_condition_tokens) != manifest["architecture"]["num_condition_tokens"]:
            raise ValueError("Loaded token count differs from bundle manifest")
        del model
    print(json.dumps({"status": "ok", "run_id": manifest["run_id"],
                      "backbone_mode": manifest["backbone"]["mode"],
                      "architecture": manifest["architecture"],
                      "strict_load": not args.check_files_only}, indent=2))


if __name__ == "__main__":
    main()
