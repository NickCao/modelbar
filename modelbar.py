import hashlib
import json
import tarfile
from dataclasses import dataclass
from functools import lru_cache

import httpx
from flask import Flask, Response, request, stream_with_context
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers

MEDIA_TYPE_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
MEDIA_TYPE_CONFIG = "application/vnd.oci.image.config.v1+json"
MEDIA_TYPE_LAYER = "application/vnd.oci.image.layer.v1.tar"

SKIP_FILES = {".gitattributes", "README.md", "LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE"}

MODEL_EXTENSIONS = {".safetensors", ".json", ".txt", ".model"}

app = Flask(__name__)
api = HfApi()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tar_header(filename: str, size: int) -> bytes:
    info = tarfile.TarInfo(name=filename)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.type = tarfile.REGTYPE
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info.tobuf(tarfile.GNU_FORMAT)


def tar_padding(size: int) -> bytes:
    remainder = size % 512
    pad = b"\0" * (512 - remainder) if remainder else b""
    return pad + b"\0" * 1024


def tar_total_size(file_size: int) -> int:
    remainder = file_size % 512
    padding = (512 - remainder) if remainder else 0
    return 512 + file_size + padding + 1024


@dataclass
class BlobEntry:
    digest: str
    tar_size: int
    filename: str
    raw_size: int
    repo: str
    revision: str
    is_lfs: bool
    data: bytes | None = None
    is_config: bool = False


@dataclass
class ResolvedModel:
    manifest_bytes: bytes
    manifest_digest: str
    blobs: dict[str, BlobEntry]


def _compute_tar_digest(repo: str, revision: str, filename: str, raw_size: int) -> str:
    """Stream a file from HF to compute the sha256 of its TAR-wrapped form."""
    header = tar_header(filename, raw_size)
    pad = tar_padding(raw_size)

    h = hashlib.sha256()
    h.update(header)

    url = hf_hub_url(repo, filename, revision=revision)
    headers = build_hf_headers()
    with httpx.stream("GET", url, headers=headers, follow_redirects=True) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
            h.update(chunk)

    h.update(pad)
    return h.hexdigest()


@lru_cache(maxsize=128)
def resolve(repo: str, revision: str) -> ResolvedModel:
    files = list(api.list_repo_tree(repo, revision=revision, recursive=True, repo_type="model"))

    blobs: dict[str, BlobEntry] = {}
    layers = []
    diff_ids = []

    for f in files:
        if not hasattr(f, "size") or f.path in SKIP_FILES:
            continue
        if not any(f.path.endswith(ext) for ext in MODEL_EXTENSIONS):
            continue

        if f.lfs:
            raw_size = f.lfs.size
            digest = _compute_tar_digest(repo, revision, f.path, raw_size)
            entry = BlobEntry(
                digest=f"sha256:{digest}",
                tar_size=tar_total_size(raw_size),
                filename=f.path,
                raw_size=raw_size,
                repo=repo,
                revision=revision,
                is_lfs=True,
            )
        else:
            url = hf_hub_url(repo, f.path, revision=revision)
            headers = build_hf_headers()
            resp = httpx.get(url, headers=headers, follow_redirects=True)
            resp.raise_for_status()
            raw_data = resp.content

            header = tar_header(f.path, len(raw_data))
            pad = tar_padding(len(raw_data))
            tar_data = header + raw_data + pad
            digest = sha256_hex(tar_data)

            entry = BlobEntry(
                digest=f"sha256:{digest}",
                tar_size=len(tar_data),
                filename=f.path,
                raw_size=len(raw_data),
                repo=repo,
                revision=revision,
                is_lfs=False,
                data=tar_data,
            )

        blobs[entry.digest] = entry
        layers.append(
            {
                "mediaType": MEDIA_TYPE_LAYER,
                "size": entry.tar_size,
                "digest": entry.digest,
                "annotations": {"org.opencontainers.image.title": entry.filename},
            }
        )
        diff_ids.append(entry.digest)

    if not layers:
        raise ValueError(f"no model files found in {repo}@{revision}")

    config = {
        "architecture": "amd64",
        "os": "linux",
        "config": {
            "Labels": {
                "org.huggingface.repo": repo,
                "org.huggingface.revision": revision,
            }
        },
        "rootfs": {"type": "layers", "diff_ids": diff_ids},
    }
    config_bytes = json.dumps(config, separators=(",", ":")).encode()
    config_digest = f"sha256:{sha256_hex(config_bytes)}"

    blobs[config_digest] = BlobEntry(
        digest=config_digest,
        tar_size=len(config_bytes),
        filename="",
        raw_size=len(config_bytes),
        repo=repo,
        revision=revision,
        is_lfs=False,
        data=config_bytes,
        is_config=True,
    )

    manifest = {
        "schemaVersion": 2,
        "mediaType": MEDIA_TYPE_MANIFEST,
        "config": {
            "mediaType": MEDIA_TYPE_CONFIG,
            "size": len(config_bytes),
            "digest": config_digest,
        },
        "layers": layers,
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
    manifest_digest = f"sha256:{sha256_hex(manifest_bytes)}"

    return ResolvedModel(
        manifest_bytes=manifest_bytes,
        manifest_digest=manifest_digest,
        blobs=blobs,
    )


_digest_to_model: dict[str, ResolvedModel] = {}


def resolve_and_cache(repo: str, revision: str) -> ResolvedModel:
    model = resolve(repo, revision)
    _digest_to_model[model.manifest_digest] = model
    return model


@app.route("/v2/")
def v2_base():
    return Response("{}", content_type="application/json")


@app.route("/v2/<path:name>/manifests/<reference>", methods=["GET", "HEAD"])
def manifests(name: str, reference: str):
    if reference.startswith("sha256:"):
        model = _digest_to_model.get(reference)
        if not model:
            return Response("manifest not found", status=404)
    else:
        revision = "main" if reference == "latest" else reference
        try:
            model = resolve_and_cache(name, revision)
        except Exception as e:
            return Response(str(e), status=404)

    headers = {
        "Content-Type": MEDIA_TYPE_MANIFEST,
        "Docker-Content-Digest": model.manifest_digest,
        "Content-Length": str(len(model.manifest_bytes)),
    }
    if request.method == "HEAD":
        return Response(headers=headers)
    return Response(model.manifest_bytes, headers=headers)


def _find_blob(digest: str) -> BlobEntry | None:
    for model in _digest_to_model.values():
        if digest in model.blobs:
            return model.blobs[digest]
    return None


def _stream_tar_blob(entry: BlobEntry):
    header = tar_header(entry.filename, entry.raw_size)
    pad = tar_padding(entry.raw_size)

    yield header

    url = hf_hub_url(entry.repo, entry.filename, revision=entry.revision)
    headers = build_hf_headers()
    with httpx.stream("GET", url, headers=headers, follow_redirects=True) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
            yield chunk

    yield pad


@app.route("/v2/<path:name>/blobs/<digest>", methods=["GET", "HEAD"])
def blobs(name: str, digest: str):
    entry = _find_blob(digest)

    if not entry:
        try:
            resolve_and_cache(name, "main")
        except Exception:
            pass
        entry = _find_blob(digest)

    if not entry:
        return Response("blob not found", status=404)

    ct = MEDIA_TYPE_CONFIG if entry.is_config else MEDIA_TYPE_LAYER
    resp_headers = {
        "Content-Type": ct,
        "Docker-Content-Digest": digest,
        "Content-Length": str(entry.tar_size),
    }

    if request.method == "HEAD":
        return Response(headers=resp_headers)

    if entry.data is not None:
        return Response(entry.data, headers=resp_headers)

    return Response(
        stream_with_context(_stream_tar_blob(entry)),
        headers=resp_headers,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HuggingFace-to-OCI registry proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
