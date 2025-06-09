#!/bin/bash

# Usage: ./validate_phase.sh <phase_number>

if [ -z "$1" ]; then
  echo "Usage: $0 <phase_number>"
  exit 1
fi

PHASE=$1
SCRIPT_DIR="$(dirname "$0")"
SCRIPT="$SCRIPT_DIR/validate_p${PHASE}.sh"

if [ ! -f "$SCRIPT" ]; then
  echo "Validation script $SCRIPT not found."
  exit 1
fi

bash "$SCRIPT" 