# Mini-swe-agent GAGAR demo

This directory owns the SWE agent server, client, SIF cache, bridge, grader,
prompts and launch scripts. It does not import or launch another example recipe.
The actor still receives the original 0/1 test outcome before group grading.

## Environment and data

Install Apptainer and use Bash 5.1 or newer on the assigned execution node.
Create a local virtualenv for the agent processes:

```bash
uv venv /local/path/mini-swe-agent-venv --python 3.12
source /local/path/mini-swe-agent-venv/bin/activate
uv pip install mini-swe-agent==2.4.1 flask pyarrow datasets swebench==4.1.0 huggingface_hub
```

From the Relax repository root, configure the model, data, cache and run paths:

```bash
export MINI_SWE_AGENT_VENV=/local/path/mini-swe-agent-venv
export MODEL_DIR=/path/to/models
export SAVE_DIR=/path/to/checkpoints
export R2E_DATA_PATH=/path/to/r2e-data
export R2E_SIF_DIR=/path/to/sif-cache
export EXP_ROOT=/path/to/new-experiment
R2E_SETUP_SKIP_SIFS=1 bash examples/gagar/minisweagent/setup_r2e_data_and_sifs.sh
RAY_NO_WAIT=1 bash scripts/entrypoint/ray-job.sh examples/gagar/minisweagent/run_gagar.sh
```

`MODEL_DIR` contains Qwen3.5-35B-A3B and, by default, Qwen3.8-27B.
`GAGAR_MODEL_PATH` can select a different grader path. The setup downloads the
R2E training/evaluation parquet data and wheels; images are converted on demand.
The submitter only submits the Ray job. `driver.sh` owns the server and trainer,
checks server readiness, forwards cancellation and waits for cleanup.

## Files

- `run_gagar.sh`: SWE training plus split GenRM; no standalone baseline or PD recipe.
- `driver.sh`: Ray job process supervision.
- `agent_server.py`, `agent_client.sh`, `agent_config.yaml`: actor execution and original judging.
- `sif_cache.py`, `setup_r2e_data_and_sifs.sh`, `check_r2e_sifs.py`: local data/image preparation.
- `group_rm.py`, `bridge.py`: existing framework hook and candidate environment routing.
- `grader.py`, `scoring.py`, `git_evidence.py`: grader adapter, weights and frozen actor changes.
- `grader.yaml`: system prompt; `instance.jinja`: task/outcome/evidence prompt.

## Grading and resource layout

This recipe keeps mini-swe-agent's original binary outcome verification and
uses Relax's `--group-rm` to request a group grader. The grader's weights feed
`compute_group_rewards`; Relax performs the final group mean subtraction.

Use the environment and data setup above, then launch this directory's `run_gagar.sh`. It uses split colocate resources: actor 8 GPUs, rollout 4 GPUs and
GenRM 4 GPUs (TP 4), with `GAGAR_MODEL_PATH` defaulting to
`${MODEL_DIR}/Qwen3.8-27B`. The SWE grader uses a 65,536-token context with
a 2,048-token reply budget per call. No deferred reward hook or custom
advantage is used.
This entry point always enables group grading; no separate SWE baseline is included.

`run_gagar.sh` enables mini-swe-agent's on-demand SIF cache by default (128 GiB,
two concurrent builds). Prepare only data and wheels with
`R2E_SETUP_SKIP_SIFS=1 bash examples/gagar/minisweagent/setup_r2e_data_and_sifs.sh`;
the first use of an image builds it into `R2E_SIF_DIR`. Configure cache scratch and registry access with the variables below. `AGENT_SERVER_SIF_CACHE=0` keeps
the prebuilt mode. The SIF lease follows the retained sandbox through outcome
verification and group grading; it is released only after the instance stops.
The cache budget excludes build scratch and cannot evict images still in use.

## Grader process boundary

`GAGAR_GRADER_COMMAND` is a trusted process command, launched in a separate
process by the agent server. It receives `--input <manifest.json>` and
`--output <result.json>`. The manifest contains the task, ordered candidates,
original outcomes, frozen pre-test patch/trajectory, original test output,
Relax GenRM service URL and a short-lived bridge capability. The result is
`{"weights": [...], "rewards": [...]}` in the same candidate order. `rewards`
is optional; corrections may change a pass to a failure, never vice versa.
SWE patches are relative to the tracked working tree after setup and before
actor execution. A private Git index freezes this baseline without changing
HEAD or the real index. This excludes pre-existing dirty image changes while
retaining actor edits, including changes to the same files. The manifest keeps
`baseline` (HEAD, tree and initial diff) and `head_patch` for audit. As in the
original harness, untracked files enter the patch only after they are staged.
Evidence capture failure still runs the original binary judge; incomplete
evidence for any passing candidate makes the group fall back to its outcomes.

The default command runs `examples.gagar.minisweagent.grader` using the mini-swe-agent
virtualenv. Its `DefaultAgent` loop calls the existing Relax GenRM `/generate`
API with complete message history. GenRM currently returns text rather than
native `tool_calls`; the recipe parses one JSON tool action per response,
without changing the framework API. `GAGAR_CONFIG` can select another grader
YAML configuration. The default budget is 40 calls and 1500 seconds.

Only `bash(rollout_n, command)` can execute model-generated commands. The
bridge resolves `rollout_n` within the current immutable group and invokes
the retained environment; it never runs those commands on the host. Missing
or invalid candidate IDs, unknown arguments and expired capabilities are
rejected. Candidate IDs are zero-based and do not encode server endpoints.
The capability is omitted from saved grader evidence after the job ends.

## Lifetime and failures

Completed training sessions transfer their sandbox to the bridge while
immediately releasing the rollout worker. Eval, infrastructure failures and
cancelled sessions keep the original cleanup path. Both passing and failing
completed candidates can be inspected. The grader cannot promote an
unsubmitted trajectory to a passing outcome.

Grading completion, failure and cancellation release the entire group.
Independent lease expiry also covers groups discarded before group RM.
Cleanup waits for an in-flight tool command and keeps failed cleanup entries
for retry; scratch is removed only after the instance stops. Repeated GETs
and release calls are safe. Original server record TTL and retained sandbox
ownership are separate.

| Environment setting         | Default | Meaning                                 |
| --------------------------- | ------- | --------------------------------------- |
| `GAGAR_CONCURRENCY`         | 4       | Concurrent grader processes             |
| `GAGAR_TIMEOUT_SECONDS`     | 1800    | Group grading budget including queueing |
| `GAGAR_RETENTION_SECONDS`   | 14400   | Maximum wait for an unclaimed group     |
| `GAGAR_MAX_RETAINED_GROUPS` | 32      | Group retention budget                  |

Retention pressure never blocks a rollout worker: a group without available
sandboxes falls back to its original outcomes. Size the retention budget and
lease duration for the configured partial-rollout / pause window. A lease
expiry loses grading for that group, not its original outcome reward.

Unavailable or malformed grading falls back for the whole group. Returned
reward fields are `score` (training), `test_reward` (original binary outcome),
`outcome_reward` (after correction), `quality_weight`, and `grader_fallback`.
Eval uses `test_reward`. Equivalent `score` values can exceed one and must not
be clipped or treated as test-pass indicators. Keep mean centering enabled
and GRPO standard-deviation normalization disabled.

## Quality grading (paper Appendix A.1)

The grader inspects all candidates together and assesses original passes on
five integer 1–5 criteria: approach (30%), precision (25%), minimality (20%),
side-effect avoidance (15%), and consistency (10%). The program computes the
initial ordering and tiers from the assessment; model output never directly
chooses arbitrary weights.

- T3 takes precedence: unrequested rewrite, test-specific workaround,
  approach=1, or both minimality and side-effect avoidance \<=2.
- T1 requires every criterion >=4, with no severe process issue or unresolved
  regression. Remaining candidates are T2.
- T1's top tied rank receives 1.0; other T1 ranks receive 0.9.
- T2 tied ranks interpolate from 0.85 to 0.4; a single tied rank receives 0.85.
- T3 receives 0.2. Tied rank counts, rather than preceding candidate counts,
  determine weights.

Rank changes require task-specific explanations. Confirmed reliance on leaked
or external solutions requires separate evidence and changes outcome to zero
before ranking and GAGAR statistics. A test-specific workaround alone is a T3
quality issue, not automatic outcome correction. The result preserves the
assessments, derived tiers and final ranking for review.

The `read_evidence` tool reads frozen pre-test patch/trajectory and original
test output in pages; `bash` inspects the current post-test sandbox. The grader
must distinguish original candidate edits from later test or inspection
changes. Reference-patch similarity is not a quality criterion.

No production training launch is part of the unit tests. Tests using fake
sandboxes and mocked GenRM transport run without Apptainer; real sandbox
validation must be explicitly run on the authorized task machine.

## SIF cache settings

| Variable                             | Default                            | Purpose                                                   |
| ------------------------------------ | ---------------------------------- | --------------------------------------------------------- |
| `AGENT_SERVER_SIF_CACHE`             | `1` in this launcher               | Use on-demand conversion; `0` selects prebuilt SIFs       |
| `AGENT_SERVER_SIF_CACHE_GB`          | `128`                              | Cached image size budget; leased images cannot be evicted |
| `AGENT_SERVER_SIF_BUILD_CONCURRENCY` | `2`                                | Concurrent image conversions                              |
| `AGENT_SERVER_SIF_BUILD_TIMEOUT`     | `1800`                             | Conversion timeout in seconds                             |
| `AGENT_SERVER_SIF_BUILD_TMP_DIR`     | agent-server scratch / `sif-build` | Build scratch, separate from the image budget             |
| `AGENT_SERVER_SIF_REGISTRY`          | unset                              | Optional registry mirror                                  |
| `AGENT_SERVER_SIF_PROXY`             | unset                              | Proxy used only for image conversion                      |

Set these variables before submission; the launcher forwards them into the Ray
job. `setup_r2e_data_and_sifs.sh` can also prebuild images when
`R2E_SETUP_SKIP_SIFS` is unset. This preserves the prebuilt and on-demand paths
without including another training recipe.
