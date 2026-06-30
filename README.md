# modelbar

A lightweight proxy that serves [HuggingFace](https://huggingface.co) model repositories as [OCI](https://opencontainers.org) artifacts, designed for use with [Kubernetes image volumes](https://kubernetes.io/docs/tasks/configure-pod-container/image-volumes/).

The proxy synthesizes OCI manifests on the fly and redirects blob requests to HuggingFace's CDN via HTTP 307 — it never stores or transfers model data itself.

## How it works

```
CRI-O / podman                   modelbar                      HuggingFace
───────────────                   ────────                      ───────────

GET /v2/                     →    200 OK

GET /v2/.../manifests/main   →    Query HF API for file list
                             ←    Synthesized OCI manifest

GET /v2/.../blobs/sha256:... →    Config blob? Serve directly
                                  Model blob?  307 redirect  →  CDN download
```

Each file in the HuggingFace repo becomes a separate OCI layer with an [`org.opencontainers.image.title`](https://github.com/opencontainers/image-spec/blob/main/annotations.md) annotation preserving the original filename.

## Usage

### Run locally

```
uv run modelbar --port 5000
```

### Run with Docker / Podman

```
podman build -t modelbar .
podman run --rm --network host modelbar
```

### Pull a model

```
podman artifact pull --tls-verify=false localhost:5000/qwen/qwen2.5-1.5b-instruct:main
```

The OCI reference maps directly to the HuggingFace repository name (lowercased). The tag corresponds to the HuggingFace branch (`main`, `fp16`, etc.).

### Use as a Kubernetes image volume

Requires Kubernetes 1.33+ / OpenShift 4.20+ with image volume support.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: inference
spec:
  containers:
    - name: vllm
      image: vllm/vllm-openai:latest
      volumeMounts:
        - name: model
          mountPath: /models
  volumes:
    - name: model
      image:
        reference: modelbar.internal/qwen/qwen2.5-1.5b-instruct:main
        pullPolicy: IfNotPresent
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Listen address |
| `--port` | `5000` | Listen port |

For private or gated repos, set the `HF_TOKEN` environment variable (used automatically by the `huggingface_hub` SDK).

## License

[MIT](LICENSE)
