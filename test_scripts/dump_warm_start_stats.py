"""Step-by-step warm-start diagnostics: monkey-patch
ContactWarmStarter.apply to read its diag counters AFTER each call,
record per-step (matched, unmatched, cold_normal, cold_friction).
"""
import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

CONFIG_PATH = "../examples/conf"


@hydra.main(config_path=CONFIG_PATH, config_name="helhest_obstacle_benchmark", version_base=None)
def main(cfg: DictConfig):
    import warp as wp
    from axion import (
        EngineConfig,
        ExecutionConfig,
        LoggingConfig,
        RenderingConfig,
        SimulationConfig,
    )
    from axion.collision.warm_start import ContactWarmStarter

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "helhest"))
    from obstacle_benchmark import HelhestObstacleBenchmark  # noqa: E402

    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = HelhestObstacleBenchmark(
        sim_config, render_config, exec_config, engine_config, logging_config,
        control_mode=cfg.control.mode,
        k_p=cfg.control.k_p, k_d=cfg.control.k_d,
        friction=cfg.friction_coeff, drive_velocity=cfg.drive_velocity,
    )

    # Monkey-patch: log diag counters after each apply().
    records = []
    orig_apply = ContactWarmStarter.apply
    def patched(self, contacts, data, dt):
        orig_apply(self, contacts, data, dt)
        if not self.enabled:
            return
        wp.synchronize()
        records.append((
            int(self._diag_attempts.numpy()[0]),
            int(self._diag_matched.numpy()[0]),
            int(self._diag_cold_normal.numpy()[0]),
            int(self._diag_cold_friction.numpy()[0]),
        ))
    ContactWarmStarter.apply = patched

    sim.run()

    R = np.array(records)  # (steps, 4)
    if len(R) == 0:
        print("No records collected — warm start disabled?")
        return
    attempts, matched, cold_n, cold_f = R.T
    unmatched = attempts - matched
    print(f"\n=== Warm-start per-step stats over {len(R)} steps ===")
    print(f"attempts       p50={int(np.percentile(attempts,50))} p95={int(np.percentile(attempts,95))} max={int(attempts.max())}")
    print(f"matched        p50={int(np.percentile(matched,50))} p95={int(np.percentile(matched,95))} max={int(matched.max())}")
    print(f"unmatched      p50={int(np.percentile(unmatched,50))} p95={int(np.percentile(unmatched,95))} max={int(unmatched.max())}")
    print(f"cold_normal    p50={int(np.percentile(cold_n,50))} p95={int(np.percentile(cold_n,95))} max={int(cold_n.max())} sum={int(cold_n.sum())}")
    print(f"cold_friction  p50={int(np.percentile(cold_f,50))} p95={int(np.percentile(cold_f,95))} max={int(cold_f.max())} sum={int(cold_f.sum())}")
    print(f"\nsteps with cold_normal>0:   {int((cold_n>0).sum())}/{len(R)}")
    print(f"steps with cold_friction>0: {int((cold_f>0).sum())}/{len(R)}")
    if (cold_n > 0).any():
        idx_first = int(np.argmax(cold_n > 0))
        print(f"\nFirst step with cold-start: step {idx_first}, "
              f"unmatched={unmatched[idx_first]}, cold_normal={cold_n[idx_first]}")

    print("\n--- per-step head (steps 0..15) ---")
    print(f"{'step':>4} {'attempts':>8} {'matched':>7} {'unmatched':>9} {'cold_n':>6} {'cold_f':>6}")
    for i in range(min(20, len(R))):
        print(f"{i:>4d} {attempts[i]:>8d} {matched[i]:>7d} {unmatched[i]:>9d} {cold_n[i]:>6d} {cold_f[i]:>6d}")
    print("\n--- mid-run (steps 60..70) ---")
    for i in range(60, min(70, len(R))):
        print(f"{i:>4d} {attempts[i]:>8d} {matched[i]:>7d} {unmatched[i]:>9d} {cold_n[i]:>6d} {cold_f[i]:>6d}")
    print("\n--- late-run (steps 150..160) ---")
    for i in range(150, min(160, len(R))):
        print(f"{i:>4d} {attempts[i]:>8d} {matched[i]:>7d} {unmatched[i]:>9d} {cold_n[i]:>6d} {cold_f[i]:>6d}")


if __name__ == "__main__":
    main()
