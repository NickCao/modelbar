import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

import httpx
from flask import Flask, Response, redirect, request
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers

MEDIA_TYPE_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
MEDIA_TYPE_EMPTY_CONFIG = "application/vnd.oci.empty.v1+json"
MEDIA_TYPE_LAYER = "application/octet-stream"
ARTIFACT_TYPE = "application/vnd.modelbar.model.v1"

EMPTY_CONFIG = b"{}"
EMPTY_CONFIG_DIGEST = f"sha256:{hashlib.sha256(EMPTY_CONFIG).hexdigest()}"

SKIP_FILES = {".gitattributes", "README.md", "LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE"}

MODEL_EXTENSIONS = {".safetensors", ".json", ".txt", ".model"}

app = Flask(__name__)
api = HfApi()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class BlobEntry:
    digest: str
    filename: str
    size: int
    repo: str
    revision: str
    is_lfs: bool
    data: bytes | None = None


@dataclass
class ResolvedModel:
    manifest_bytes: bytes
    manifest_digest: str
    blobs: dict[str, BlobEntry]


@lru_cache(maxsize=128)
def resolve(repo: str, revision: str) -> ResolvedModel:
    files = list(api.list_repo_tree(repo, revision=revision, recursive=True, repo_type="model"))

    blobs: dict[str, BlobEntry] = {}
    layers = []

    for f in files:
        if not hasattr(f, "size") or f.path in SKIP_FILES:
            continue
        if not any(f.path.endswith(ext) for ext in MODEL_EXTENSIONS):
            continue

        if f.lfs:
            entry = BlobEntry(
                digest=f"sha256:{f.lfs.sha256}",
                filename=f.path,
                size=f.lfs.size,
                repo=repo,
                revision=revision,
                is_lfs=True,
            )
        else:
            url = hf_hub_url(repo, f.path, revision=revision)
            headers = build_hf_headers()
            resp = httpx.get(url, headers=headers, follow_redirects=True)
            resp.raise_for_status()
            data = resp.content
            digest = sha256_hex(data)
            entry = BlobEntry(
                digest=f"sha256:{digest}",
                filename=f.path,
                size=len(data),
                repo=repo,
                revision=revision,
                is_lfs=False,
                data=data,
            )

        blobs[entry.digest] = entry
        layers.append(
            {
                "mediaType": MEDIA_TYPE_LAYER,
                "size": entry.size,
                "digest": entry.digest,
                "annotations": {"org.opencontainers.image.title": entry.filename},
            }
        )

    if not layers:
        raise ValueError(f"no model files found in {repo}@{revision}")

    blobs[EMPTY_CONFIG_DIGEST] = BlobEntry(
        digest=EMPTY_CONFIG_DIGEST,
        filename="",
        size=len(EMPTY_CONFIG),
        repo=repo,
        revision=revision,
        is_lfs=False,
        data=EMPTY_CONFIG,
    )

    manifest = {
        "schemaVersion": 2,
        "mediaType": MEDIA_TYPE_MANIFEST,
        "artifactType": ARTIFACT_TYPE,
        "config": {
            "mediaType": MEDIA_TYPE_EMPTY_CONFIG,
            "size": len(EMPTY_CONFIG),
            "digest": EMPTY_CONFIG_DIGEST,
        },
        "layers": layers,
        "annotations": {
            "org.huggingface.repo": repo,
            "org.huggingface.revision": revision,
        },
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

    if entry.data is not None:
        headers = {
            "Content-Type": MEDIA_TYPE_LAYER,
            "Docker-Content-Digest": digest,
            "Content-Length": str(len(entry.data)),
        }
        if request.method == "HEAD":
            return Response(headers=headers)
        return Response(entry.data, headers=headers)

    url = hf_hub_url(entry.repo, entry.filename, revision=entry.revision)
    headers = build_hf_headers()
    resp = httpx.head(url, headers=headers, follow_redirects=True)
    return redirect(str(resp.url), code=307)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HuggingFace-to-OCI registry proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
