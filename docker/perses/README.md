# Perses + Prometheus — vLLM observability stack

`mjolnir perses up|down|status|logs` drives this stack (docker compose under
the hood). It is **independent of the vLLM server**, on the same terms as
`docker/litellm/`: Prometheus scrapes the server's `/metrics` through
`host.docker.internal:<port>` (the resolved port, same as `mjolnir serve`),
so the stack works whether the server is up or down — while the server is
down the `vllm` target simply shows `down`.

Services (compose project `mjolnir-perses`):

| service      | image                  | host port (default) | role |
|--------------|------------------------|---------------------|------|
| `perses`     | `persesdev/perses`     | `8080`              | the dashboard UI (the OSS observability dashboard) |
| `prometheus` | `prom/prometheus`      | `9090`              | scrapes the vLLM `/metrics` every 15s (targets view: `/targets`) |

The UI: **http://127.0.0.1:8080** — the two vLLM dashboards
(*Performance Statistics*, *Query Statistics*) are already there.

## The vLLM dashboards — vendored from vllm-project/vllm

`dashboards/` vendors `examples/observability/dashboards/perses/`
(fetched 2026-09-30) from the vLLM repo. Three adaptations make them work
standalone (every panel/query/layout is otherwise verbatim):

1. **`kind: PersesDashboard` → `kind: Dashboard`** — the upstream files use
   the k8s-operator CRD kind form; the standalone API/provisioning only
   accepts `Dashboard` (`pkg/model/api/v1/kind.go::GetKind` has no
   `PersesDashboard` case).
2. **`project: ""` → `project: vllm`** — standalone Perses is
   project-centric: the UI only renders a dashboard whose `metadata.project`
   matches an existing `Project` resource (it groups dashboards by project
   and drops the rest). The upstream files ship `project: ""`, which is
   invisible — so `project.yaml` provisions a `vllm` project and both
   dashboards point at it.
3. **The `namespace=`/`service=` matchers are satisfied by target labels.**
   Every `query_statistics` panel carries `namespace="$NS",service="$SVC"`
   label matchers (and the `$NS`/`$SVC` variables query `up{service=~".*vllm.*"}`).
   Those labels don't exist on a plain Prometheus scrape — so
   `prometheus.template.yaml` attaches `namespace: mjolnir` +
   `service: mjolnir-vllm` to the scrape job's target, which makes
   `up{...}` and every panel matcher line up. (The `performance_statistics`
   dashboard instead filters on the `model_name` label, which vLLM exports
   by default.)

The Prometheus datasource itself (`global-datasource.yaml`) is provisioned
the same way — the name `accelerators-thanos-querier-datasource` is fixed by
the upstream dashboard files (the k8s-operator example they derive from);
the URL points at the compose-network `prometheus` service.

Provisioning is built into Perses (`provisioning.folders` in `perses.yml`):
at startup, and re-synced hourly, the server loads every known resource from
the read-only `dashboards/` folder. No `percli apply` step.

## vLLM metrics — which flags are needed

**None.** Every metric both dashboards query
(`e2e_request_latency_seconds`, `time_to_first_token_seconds`,
`inter_token_latency_seconds`, `iteration_tokens_total`,
`kv_cache_usage_perc`, `num_requests_running`, `num_requests_waiting`,
`prompt_tokens_total`, `generation_tokens_total`, `request_success_total`)
is part of the **default** vLLM `/metrics` export — verified against the
live vLLM 0.30.0 server (the `mjolnir-vllm` container) in this repo.
Prometheus is on by default (the serve line never passes
`--disable-metrics`), and the dashboards filter by the `model_name` label,
which vLLM labels every request metric with (the served model name). The
existing serve flags `--enable-force-include-usage` /
`--enable-prompt-tokens-details` affect response bodies, not metrics — left
as-is.

## Data layout

All user-persisted files in one dir — `$MJOLNIR_PERSER_DATA` or
`$MJOLNIR_DATA/perses` (default `~/.local/share/mjolnir/perses`):

```
perses.env          optional env file ($MJOLNIR_PERSER_ENV): ports, image tags,
                    PERSER_VLLM_TARGET (override the scrape target)
prometheus.yml      rendered on every `perses up` (the scrape target)
db/                 Perses file database (project + dashboards + datasource)
prom/               Prometheus TSDB (scrape history survives down/up)
```

`perses down` keeps all of it; `perses down --purge` wipes `db/` + `prom/`.
The services run as their images' nonroot uid, so the subdirs they create
under `db/`/`prom/` are `0750` and undeletable by the host user — the purge
therefore runs a throwaway **root** `busybox` container to `rm -rf` them
(root bypasses the permissions), falling back to a host `rmtree` if docker
can't run (see `_purge_data` in `src/mjolnir-cli/mjolnir_cli/stack.py`).
