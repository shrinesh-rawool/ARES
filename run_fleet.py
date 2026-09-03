"""
run_fleet.py — Fleet orchestrator.

Launches 3 independent AMR node processes and the Pygame visualizer,
then performs periodic health checks. Clean shutdown on Ctrl+C.

Node layout (crisscross paths to force conflict resolution)
-----------------------------------------------------------
  AMR 1 : (1,1)  → (18,18)   urgency=1
  AMR 2 : (18,1) → (1,18)    urgency=2   ← higher priority
  AMR 3 : (1,18) → (18,1)    urgency=1
"""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

NODE_CONFIGS = [
    # (id,  start,    goal,    urgency)
    # AMR 1: left → right along y=14 corridor (crosses AMR 2 at (14,14))
    ("1", "2,14",  "27,14", "1"),
    # AMR 2: top → bottom along x=14 corridor (crosses AMR 1 at (14,14)) — higher urgency
    ("2", "14,2",  "14,27", "2"),
    # AMR 3: bottom-left → top-right diagonal
    ("3", "2,27",  "27,2",  "1"),
    # AMR 4: top-left → bottom-right diagonal
    ("4", "2,2",   "27,27", "1"),
]

ZMQ_SETTLE_SECS   = 1.5   # wait after launching nodes before the visualizer
HEALTH_CHECK_SECS = 2.0   # interval for checking node health


def launch_node(node_id: str, start: str, goal: str, urgency: str, policy: str = "rl") -> subprocess.Popen:
    cmd = [
        sys.executable, "node.py",
        "--id",      node_id,
        "--start",   start,
        "--goal",    goal,
        "--urgency", urgency,
        "--policy",  policy,
    ]
    return subprocess.Popen(cmd, cwd=ROOT)


def launch_visualizer() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "visualizer.py"], cwd=ROOT)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AMR Fleet Orchestrator")
    parser.add_argument("--policy", default="rl", choices=["rl", "planner"], help="Policy: rl or planner")
    args = parser.parse_args()

    print("=" * 54)
    print(f"  AMR Fleet MVP — Multi-Agent Robot System [{args.policy.upper()}]")
    print("=" * 54)

    processes: list = []  # list of (label, Popen)

    # ── Launch AMR nodes ────────────────────────────────────────────
    print(f"\n[Fleet] Launching AMR edge nodes with policy: {args.policy.upper()}…")
    for node_id, start, goal, urgency in NODE_CONFIGS:
        p = launch_node(node_id, start, goal, urgency, policy=args.policy)
        processes.append((f"AMR-{node_id}", p))
        print(f"  AMR {node_id}  start={start:6s}  goal={goal:6s}  urgency={urgency}  policy={args.policy}  pid={p.pid}")

    # ── Wait for ZMQ mesh to stabilise ──────────────────────────────
    print(f"\n[Fleet] Waiting {ZMQ_SETTLE_SECS}s for ZMQ mesh to settle…")
    time.sleep(ZMQ_SETTLE_SECS)

    # ── Launch visualizer ────────────────────────────────────────────
    print("[Fleet] Launching visualizer…")
    vis = launch_visualizer()
    processes.append(("Visualizer", vis))
    print(f"  Visualizer pid={vis.pid}")
    print("\n[Fleet] All processes running.  Press Ctrl+C to stop.\n")

    # ── Health-check loop ────────────────────────────────────────────
    try:
        while True:
            time.sleep(HEALTH_CHECK_SECS)

            # If visualizer exits, user closed the window → shut down fleet
            if vis.poll() is not None:
                print("[Fleet] Visualizer closed. Shutting down fleet…")
                break

            # Warn if any node process crashed
            for label, p in processes:
                if label == "Visualizer":
                    continue
                ret = p.poll()
                if ret is not None:
                    print(f"[Fleet] WARNING: {label} exited unexpectedly (code {ret}).")

    except KeyboardInterrupt:
        print("\n[Fleet] Ctrl+C received. Shutting down…")

    finally:
        for label, p in processes:
            if p.poll() is None:
                p.terminate()
                print(f"  Terminated {label} (pid={p.pid})")

        # Grace period, then force-kill stragglers
        time.sleep(0.8)
        for label, p in processes:
            if p.poll() is None:
                p.kill()
                print(f"  Force-killed {label} (pid={p.pid})")

        print("[Fleet] Shutdown complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()