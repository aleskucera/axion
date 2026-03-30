#!/usr/bin/env bash
# Patch MJX solver.py to replace jax.lax.while_loop with _while_loop_scan
# so that reverse-mode AD (jax.grad) works through the constraint solver.
#
# Usage: bash scripts/patch_mjx_solver.sh

set -e

SOLVER=$(python -c "import mujoco.mjx._src.solver as m; print(m.__file__)")
echo "Patching: $SOLVER"

# Check if already patched
if grep -q "_while_loop_scan" "$SOLVER"; then
    REMAINING=$(grep -c "jax\.lax\.while_loop" "$SOLVER" || true)
    if [ "$REMAINING" -eq 0 ]; then
        echo "Already patched — nothing to do."
        exit 0
    fi
fi

# Verify _while_loop_scan is defined in the file
if ! grep -q "def _while_loop_scan" "$SOLVER"; then
    echo "ERROR: _while_loop_scan not defined in $SOLVER — cannot patch."
    exit 1
fi

cp "$SOLVER" "${SOLVER}.bak"
echo "Backup saved to ${SOLVER}.bak"

sed -i \
    's/jax\.lax\.while_loop(cond, body, ls_ctx)/_while_loop_scan(cond, body, ls_ctx, m.opt.ls_iterations)/g' \
    "$SOLVER"

sed -i \
    's/jax\.lax\.while_loop(cond, body, ctx)/_while_loop_scan(cond, body, ctx, m.opt.iterations)/g' \
    "$SOLVER"

REMAINING=$(grep -c "jax\.lax\.while_loop" "$SOLVER" || true)
if [ "$REMAINING" -gt 0 ]; then
    echo "WARNING: $REMAINING jax.lax.while_loop call(s) still remain — check manually."
else
    echo "Patch applied successfully."
fi
