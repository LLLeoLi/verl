#!/bin/bash
# ==============================================================================
# Lightweight "train script" stand-in for verifying the JD/ECP Ray bootstrap.
#
# Usage (platform runs this on EVERY node, same as a real train script):
#   bash aiic_recipe/start_ray_jd.sh aiic_recipe/sanity_ray.sh
#
# start_ray_jd.sh does the head/worker Ray rendezvous, then `ray job submit`s
# this script on the head. We just run the cluster self-check payload -- no
# model load, no training. Exit code propagates so you can gate a real run:
#   bash aiic_recipe/start_ray_jd.sh aiic_recipe/sanity_ray.sh && echo LAUNCH-OK
# ==============================================================================
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python3 "${SCRIPT_DIR}/sanity_ray.py" "$@"
