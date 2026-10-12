"""Download the official 512px VGGT-Omega checkpoint and verify its checksum."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path(os.environ.get("OMEGA_ROOT", "third_party/vggt-omega")) / "ckpt"
    )
    parser.add_argument("--revision", default="1041e80fc0e911235d3426b0a3d9a81075111a53")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    repository = "facebook/VGGT-Omega"
    filename = "vggt_omega_1b_512.pt"
    info = HfApi().model_info(repository, revision=args.revision, files_metadata=True)
    item = next(file for file in info.siblings if file.rfilename == filename)
    metadata = dict(
        repository=repository,
        revision=info.sha,
        filename=filename,
        source_url=f"https://huggingface.co/{repository}/blob/{info.sha}/{filename}",
        expected_bytes=item.size,
        expected_sha256=item.lfs.sha256,
    )
    metadata_path = args.out / "vggt_omega_1b_512.download.json"
    try:
        print(json.dumps(dict(phase="download", **metadata)), flush=True)
        path = Path(hf_hub_download(repo_id=repository, filename=filename, revision=info.sha, local_dir=args.out))
    except Exception as error:
        response = getattr(error, "response", None)
        metadata.update(
            status="download_failed", error_type=type(error).__name__, http_status=getattr(response, "status_code", None)
        )
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        # Report error type/status without printing authorization headers or URLs.
        print(
            json.dumps(
                dict(status=metadata["status"], error_type=metadata["error_type"], http_status=metadata["http_status"])
            ),
            flush=True,
        )
        raise SystemExit(1) from error
    print(json.dumps(dict(phase="checksum", path=str(path))), flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    metadata.update(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())
    metadata["checksum_matches"] = (
        metadata["sha256"] == metadata["expected_sha256"] and metadata["bytes"] == metadata["expected_bytes"]
    )
    metadata["status"] = "verified" if metadata["checksum_matches"] else "checksum_failed"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if not metadata["checksum_matches"]:
        raise SystemExit("Downloaded checkpoint does not match the official file metadata.")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
