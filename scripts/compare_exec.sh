#!/bin/bash
# Usage: ./scripts/compare_exec.sh <instruction_module>
set -euo pipefail

INSTR="${1:?Usage: $0 <instruction_module>}"
cd "$(dirname "$0")/.."
cargo expand -p tracer --lib "instruction::${INSTR}" 2>/dev/null \
    | cargo run -q -p extract-fn -- exec ast_exec
