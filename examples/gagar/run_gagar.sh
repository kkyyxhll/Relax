#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
: "${MODEL_DIR:?}"
: "${MINI_SWE_AGENT_VENV:?}"
export ENABLE_GAGAR=1
export AGENT_SERVER_SIF_CACHE="${AGENT_SERVER_SIF_CACHE:-1}"
export AGENT_SERVER_SIF_CACHE_GB="${AGENT_SERVER_SIF_CACHE_GB:-128}"
export AGENT_SERVER_SIF_BUILD_CONCURRENCY="${AGENT_SERVER_SIF_BUILD_CONCURRENCY:-2}"
export GAGAR_GRADER_COMMAND="${GAGAR_GRADER_COMMAND:-\"${MINI_SWE_AGENT_VENV}/bin/python\" -m examples.gagar.grader}"
export GAGAR_MODEL_PATH="${GAGAR_MODEL_PATH:-${MODEL_DIR}/Qwen3.8-27B}"
export PYTHONPATH="${RELAX_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec bash "${RELAX_DIR}/examples/mini_swe_agent/run_mini_swe_agent.sh"
