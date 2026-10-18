# Releasing

**Story:** "I want to publish what is on `main` — the two Python packages, the
Docker image and the interactive site — with one click, and never hand-edit a
version number."

## What gets released, and how it is versioned

| artifact | where | version | tag |
|---|---|---|---|
| `mjolnir-kernel` | [PyPI](https://pypi.org/project/mjolnir-kernel/) | its own semver, from its tag history | `mjolnir-kernel-vX.Y.Z` |
| `mjolnir-cli` | [PyPI](https://pypi.org/project/mjolnir-cli/) | its own semver, from its tag history | `mjolnir-cli-vX.Y.Z` |
| `narandill/mjolnir-vllm` | [Docker Hub](https://hub.docker.com/r/narandill/mjolnir-vllm) | the **pinned upstream vLLM** (`VLLM_VERSION` = the Dockerfile's `BASE_IMAGE`) | `:v0.32.0`, `:v0.32.0-<sha>`, `:latest` |
| the release | [GitHub Releases](https://github.com/thor-arena/mjolnir/releases) | the date | `release-YYYY.MM.DD[.N]` |
| the interactive site | GitHub Pages | — | built from the release tag |

The tag is the source of truth: the publishing workflows stamp `__version__`
from the tag into the package before they build, so a release never needs a
version-bump commit on `main`.

## Cut a release

**Actions → Release → Run workflow** (on `main`):

| input | default | meaning |
|---|---|---|
| `bump` | `patch` | `patch` / `minor` / `major` for each package that is released |
| `packages` | `changed` | `changed` releases a package only if its source changed since its last tag (`src/mjolnir/` for the kernel; `src/mjolnir-cli/` plus everything its wheel bundles — `configs/`, `docker/litellm`, `docker/perses`, `scripts/hw` — for the CLI); `both` / `cli` / `kernel` / `none` force it |
| `image` | on | build and push the Docker image (switch it off for a CLI-only release: the image does not ship the CLI) |
| `pages` | on | rebuild the interactive site |
| `dry_run` | off | plan + checks only — see what *would* be released |

The run:

1. **plans** with [`scripts/release_plan.py`](../../scripts/release_plan.py):
   the next version per package = its latest tag bumped (or the version in its
   `__init__.py` when that is higher — a deliberate hand bump — or when no tag
   exists yet: the first release);
2. **checks** — the same lint, tests, README reproducibility and package-build
   gates as a pull request ([`ci.yml`](../../.github/workflows/ci.yml));
3. **tags** the commit the checks ran on and creates the GitHub Release
   (below) — nothing is committed to `main`, which stays protected;
4. **publishes** by dispatching the tag-driven workflows on the new tags:
   [`release-kernel.yml`](../../.github/workflows/release-kernel.yml),
   [`release-cli.yml`](../../.github/workflows/release-cli.yml) (PyPI, trusted
   publishing), [`docker-image.yml`](../../.github/workflows/docker-image.yml)
   (Docker Hub) and [`pages.yml`](../../.github/workflows/pages.yml).

Preview locally: `python3 scripts/release_plan.py --bump minor --packages changed`
(the plan) and `python3 scripts/release_notes.py --vllm 0.31.0 --cli-version 0.2.1
--cli-previous mjolnir-cli-v0.2.0 --previous-umbrella release-2026.10.15` (the
notes). A dry run shows both in the plan job's summary.

## The release notes

One GitHub Release per run, on the umbrella tag, titled with only what it
ships (`Mjolnir 2026.10.20 — mjolnir-kernel 0.1.1 · image v0.31.0`).
[`scripts/release_notes.py`](../../scripts/release_notes.py) writes the body:

- **Install** lines and a table of all three artifacts (unreleased ones as
  `unchanged (<version>)`);
- one section per shipped artifact, each listing what changed in the files
  that artifact is built from, since that artifact's own previous release:

| section | paths | since |
|---|---|---|
| `mjolnir-kernel X.Y.Z` | `src/mjolnir` | its last `mjolnir-kernel-v*` tag |
| `mjolnir-cli X.Y.Z` | `src/mjolnir-cli`, `configs`, `docker/litellm`, `docker/perses`, `scripts/hw` | its last `mjolnir-cli-v*` tag |
| `image vX.Y.Z` | what the Dockerfile copies: `Dockerfile`, `patches/`, the patch and check scripts, `LICENSE`, `NOTICE` (not `README.md` / `PATCHES.md`) | the previous umbrella tag |

An entry is a pull request — its title and `#number`, from the merge commit
on `main` — or a commit pushed to `main` directly. Dependabot PRs and the
branch commits inside a PR are left out. A PR that touched two artifacts
appears under both. The image section also says which vLLM it is built on
(and flags a base bump) and which kernel it ships. Kernel changes appear once,
under the kernel; if the kernel is not released but changed, the image
section warns that it ships unreleased kernel code.

PyPI users find the notes through each project's *Changelog* link (the
GitHub Releases page).

## When a run fails

Re-running the Release workflow is safe: a tag that already exists on the
same commit is kept, and an existing release gets its notes rewritten. The
plan then sees those tags as the latest ones, though, so it releases nothing
new. To retry a failed publish, dispatch that publisher on its tag instead:
`gh workflow run release-kernel.yml --ref mjolnir-kernel-v0.1.1` (or
`release-cli.yml`, `docker-image.yml -f push=true`, `pages.yml`).

Pushing a `mjolnir-kernel-v*`, `mjolnir-cli-v*` or `release-*` tag by hand
runs the matching workflow too — the manual workflow only saves the
bookkeeping.

## One-time setup

1. **PyPI** — for each project add a trusted publisher (a *pending* one before
   the first upload): owner `thor-arena`, repository `mjolnir`, workflow
   `release-kernel.yml` / `release-cli.yml`, environment `pypi-kernel` /
   `pypi-cli`.
2. **GitHub → Settings → Environments** — create `pypi-kernel` and `pypi-cli`
   (deployment tags `mjolnir-kernel-v*` / `mjolnir-cli-v*`; a required
   reviewer is a good extra gate), and allow the tag pattern `release-*` on
   `github-pages`.
3. **GitHub → Settings → Secrets and variables → Actions** — secrets
   `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` (a Docker Hub token with write
   access); optional variables `IMAGE_REPO`, `IMAGE_RUNNER`.
4. **GitHub → Settings → Pages** — source: *GitHub Actions*. With *Deploy
   from a branch* GitHub serves its own Jekyll render of `README.md` as the
   site (static PNGs, unrendered LaTeX) instead of the interactive charts;
   `pages.yml` fails with that message until it is switched.
5. **GitHub → Settings → Rules** — a tag ruleset for `mjolnir-*-v*` and
   `release-*` (block updates and deletions), and a branch ruleset for `main`
   that requires the CI checks on pull requests.
6. **GitHub → Settings → Actions → General** — workflow permissions: *Read
   repository contents*; the Release workflow asks for `contents: write`
   (tags, the release) and `actions: write` (dispatch) itself.
