#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Run one grader session. Environment provisioning stays with the task worker.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
: "${MINI_SWE_AGENT_VENV:?Set the mini-swe-agent virtualenv path}"
exec "${MINI_SWE_AGENT_VENV}/bin/python" -m examples.gagar.core.grs "$@"
