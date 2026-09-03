"""
visualize_training.py — Interactive Live Multi-Agent Training Visualizer (4 AMRs).

Renders the multi-agent DQN training process in real time on a live Pygame dashboard:
* Simulates 4 autonomous AMRs navigating and learning simultaneously in the same warehouse
* Displays active layout (6 scenarios: aisles, chokepoints, pillars, etc.)
* Renders each robot in distinct colors (Green, Cyan, Purple, Orange) with individual trails and goals
* Real-time Q-Value horizontal bar chart showing neural network confidence for any focused robot (Press 1–4)
* Running fleet performance metrics (success rate %, inter-agent collisions, static collisions)
* Interactive controls: Pause [SPC], Fast-Forward [F], Explore/Eval [E], Next Layout [N], Focus Robot [1-4], Save [S]
"""

import math
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pygame
import torch

from core.grid import WarehouseGrid
from core.layouts import LAYOUT_NAMES, get_layout
from core.rl_agent import DQNAgent
from core.rl_env import ACTIONS, AMRWarehouseEnv, LOCAL_RADIUS, OBS_DIM

# ─────────────────────────────────────────────────────────────────────
# Layout & Window Constants
# ─────────────────────────────────────────────────────────────────────

GRID_SIZE = 30
CELL_SIZE = 24  # 30 * 24 = 720 px
PANEL_W = 320
SCREEN_W = GRID_SIZE * CELL_SIZE + PANEL_W  # 1040 px
SCREEN_H = GRID_SIZE * CELL_SIZE  # 720 px

ACTION_LABELS = ["North (↑)", "South (↓)", "West (←)", "East (→)", "Wait (⏸)"]

# Distinct Robot Colors for 4 AMRs
ROBOT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "1": (70, 220, 130),   # Emerald Green
    "2": (40, 200, 255),   # Cyan Blue
    "3": (200, 100, 255),  # Electric Violet
    "4": (255, 160, 50),   # Amber Orange
}

C = {
    "bg": (18, 18, 26),
    "grid_line": (34, 34, 46),
    "shelf": (110, 65, 60),
    "obstacle": (220, 60, 60),
    "panel_bg": (12, 12, 18),
    "divider": (45, 45, 62),
    "text": (220, 220, 235),
    "text_dim": (120, 120, 145),
    "sensor_box": (100, 180, 255, 35),
    "bar_bg": (35, 35, 48),
    "bar_fill": (60, 140, 240),
    "bar_best": (50, 220, 120),
    "accent": (255, 180, 50),
    "warn": (255, 80, 80),
}


def cell_rect(x: int, y: int) -> Tuple[int, int, int, int]:
    return (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)


def cell_center(x: int, y: int) -> Tuple[int, int]:
    return (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Live Multi-Agent RL Training Visualizer")
    parser.add_argument("--layout", type=int, default=0, help="Initial layout index (0-5)")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Starting exploration epsilon")
    parser.add_argument("--eval", action="store_true", help="Pure evaluation mode (epsilon=0.0)")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of AMRs (1-4)")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("AMR Fleet Multi-Agent RL — Live Training Visualizer (4 AMRs)")

    font_title = pygame.font.SysFont("monospace", 15, bold=True)
    font_body = pygame.font.SysFont("monospace", 12)
    font_bold = pygame.font.SysFont("monospace", 12, bold=True)
    font_sm = pygame.font.SysFont("monospace", 10)
    clock = pygame.time.Clock()

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "amr_rl_policy.pt")

    # Initialize 4-Agent Environment & Shared DQN Agent
    num_agents = max(1, min(4, args.num_agents))
    grid = WarehouseGrid()
    env = AMRWarehouseEnv(grid=grid, max_steps=120, num_agents=num_agents)
    agent = DQNAgent(obs_dim=OBS_DIM, num_actions=5, device="cpu")

    start_eps = 0.0 if args.eval else args.epsilon
    if os.path.exists(model_path):
        agent.load(model_path, epsilon=start_eps)
        print(f"[Visualizer] Loaded existing model from {model_path} (epsilon={agent.epsilon:.2f})")
    else:
        agent.epsilon = start_eps

    current_layout_idx = args.layout % len(LAYOUT_NAMES)
    layout_name = env.set_layout(current_layout_idx)

    obs_dict = env.reset(randomize_obstacles=True, layout=current_layout_idx)
    episode = 1
    ep_steps = 0
    ep_rewards = {aid: 0.0 for aid in env.agent_ids}
    ep_collisions = {aid: 0 for aid in env.agent_ids}
    ep_peer_collisions = 0
    trails = {aid: [env.positions[aid]] for aid in env.agent_ids}

    focused_agent_id = "1"

    # Fleet metrics history
    recent_fleet_successes: List[bool] = []
    recent_peer_collisions: List[int] = []

    paused = False
    fast_mode = False
    status_text = "TRAINING (4 AMRs)" if agent.epsilon > 0.05 else "EVALUATION (4 AMRs)"
    status_color = C["accent"] if agent.epsilon > 0.05 else ROBOT_COLORS["1"]

    running = True
    while running:
        # ── 1. Event Handling & Hotkeys ──────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_f:
                    fast_mode = not fast_mode

                elif event.key == pygame.K_e:
                    if agent.epsilon > 0.05:
                        agent.epsilon = 0.0
                        status_text = "EVAL MODE (GREEDY)"
                        status_color = ROBOT_COLORS["1"]
                    else:
                        agent.epsilon = 0.35
                        status_text = "EXPLORE MODE (EPS=0.35)"
                        status_color = C["accent"]

                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    requested_id = chr(event.key)
                    if requested_id in env.agent_ids:
                        focused_agent_id = requested_id

                elif event.key == pygame.K_n:
                    current_layout_idx = (current_layout_idx + 1) % len(LAYOUT_NAMES)
                    layout_name = env.set_layout(current_layout_idx)
                    obs_dict = env.reset(randomize_obstacles=True, layout=current_layout_idx)
                    trails = {aid: [env.positions[aid]] for aid in env.agent_ids}
                    ep_rewards = {aid: 0.0 for aid in env.agent_ids}
                    ep_collisions = {aid: 0 for aid in env.agent_ids}
                    ep_peer_collisions = 0
                    ep_steps = 0

                elif event.key == pygame.K_r:
                    obs_dict = env.reset(randomize_obstacles=True, layout=current_layout_idx, randomize_routes=True)
                    trails = {aid: [env.positions[aid]] for aid in env.agent_ids}
                    ep_rewards = {aid: 0.0 for aid in env.agent_ids}
                    ep_collisions = {aid: 0 for aid in env.agent_ids}
                    ep_peer_collisions = 0
                    ep_steps = 0
                    status_text = "REROLLED RANDOM ROUTES"
                    status_color = C["accent"]

                elif event.key == pygame.K_s:
                    agent.save(model_path)
                    status_text = "SAVED MODEL TO DISK"
                    status_color = ROBOT_COLORS["1"]

        # ── 2. Run Multi-Agent RL Training Step ──────────────────────
        steps_per_frame = 12 if fast_mode else (0 if paused else 1)

        for _ in range(steps_per_frame):
            # Select actions for all active agents using shared Q-network
            actions_dict: Dict[str, int] = {}
            for aid in env.agent_ids:
                if not env.dones.get(aid, False):
                    actions_dict[aid] = agent.select_action(obs_dict[aid])
                else:
                    actions_dict[aid] = 4  # Wait if goal already reached

            next_obs_dict, rewards_dict, dones_dict, infos_dict = env.step(actions_dict)

            # Push experiences for all agents into shared memory
            for aid in env.agent_ids:
                agent.memory.push(
                    obs_dict[aid],
                    actions_dict[aid],
                    rewards_dict[aid], # type: ignore
                    next_obs_dict[aid],
                    dones_dict[aid],  # type: ignore
                )
                ep_rewards[aid] += rewards_dict[aid] # type: ignore
                if infos_dict[aid]["collision"]:
                    ep_collisions[aid] += 1
                if infos_dict[aid]["peer_collision"]:
                    ep_peer_collisions += 1

                trails[aid].append(env.positions[aid])
                if len(trails[aid]) > 35:
                    trails[aid].pop(0)

            # Gradient update
            agent.update(batch_size=64)

            obs_dict = next_obs_dict
            ep_steps += 1

            # Episode ends when all agents reach goals or timeout
            all_done = all(dones_dict.values()) # type: ignore
            if all_done:
                fleet_success = all(infos_dict[aid]["success"] for aid in env.agent_ids)
                recent_fleet_successes.append(fleet_success)
                recent_peer_collisions.append(ep_peer_collisions)

                if fleet_success:
                    status_text = "ALL 4 AMRs REACHED GOALS!"
                    status_color = ROBOT_COLORS["1"]
                elif ep_steps >= env.max_steps:
                    status_text = "TIMED OUT (HORIZON MAX)"
                    status_color = C["text_dim"]
                else:
                    status_text = "EPISODE TERMINATED"
                    status_color = C["warn"]

                episode += 1
                if episode % 10 == 0:
                    agent.update_target_network()

                # Automatically cycle layout every 5 episodes
                if episode % 5 == 0:
                    current_layout_idx = (current_layout_idx + 1) % len(LAYOUT_NAMES)
                    layout_name = env.set_layout(current_layout_idx)

                # Reset for next episode
                obs_dict = env.reset(randomize_obstacles=True, layout=current_layout_idx)
                trails = {aid: [env.positions[aid]] for aid in env.agent_ids}
                ep_rewards = {aid: 0.0 for aid in env.agent_ids}
                ep_collisions = {aid: 0 for aid in env.agent_ids}
                ep_peer_collisions = 0
                ep_steps = 0
                break

        # Compute Q-values for focused robot
        q_values_np = np.zeros(5)
        best_action = 0
        with torch.no_grad():
            focused_obs = obs_dict[focused_agent_id]
            st_t = torch.FloatTensor(focused_obs).unsqueeze(0).to(agent.device)
            q_tensor = agent.policy_net(st_t)
            q_values_np = q_tensor.cpu().numpy()[0]
            best_action = int(np.argmax(q_values_np))

        # ── 3. Render Warehouse Grid ─────────────────────────────────
        screen.fill(C["bg"])

        # Grid cells
        for gx in range(GRID_SIZE):
            for gy in range(GRID_SIZE):
                rect = cell_rect(gx, gy)
                if (gx, gy) in env.grid.static_obstacles:
                    pygame.draw.rect(screen, C["shelf"], rect)
                elif (gx, gy) in env.grid._dynamic:
                    pygame.draw.rect(screen, C["obstacle"], rect)
                pygame.draw.rect(screen, C["grid_line"], rect, 1)

        # Draw motion trails for all 4 AMRs
        for aid in env.agent_ids:
            col = ROBOT_COLORS[aid]
            for i, (tx, ty) in enumerate(trails[aid]):
                tcx, tcy = cell_center(tx, ty)
                alpha_ratio = (i + 1) / max(1, len(trails[aid]))
                radius = max(2, int(3 * alpha_ratio))
                pygame.draw.circle(screen, col, (tcx, tcy), radius)

        # Draw target goals for all 4 AMRs
        pulse = int(3 * math.sin(time.time() * 6))
        for aid in env.agent_ids:
            col = ROBOT_COLORS[aid]
            gx, gy = env.goals[aid]
            gcx, gcy = cell_center(gx, gy)
            pygame.draw.circle(screen, col, (gcx, gcy), CELL_SIZE // 2 + pulse, 2)
            pygame.draw.circle(screen, col, (gcx, gcy), 4)
            lbl_g = font_sm.render(f"G{aid}", True, col)
            screen.blit(lbl_g, (gcx + 8, gcy - 8))

        # Draw 5x5 sensor footprint for focused robot
        frx, fry = env.positions[focused_agent_id]
        sensor_rect = (
            (frx - LOCAL_RADIUS) * CELL_SIZE,
            (fry - LOCAL_RADIUS) * CELL_SIZE,
            (2 * LOCAL_RADIUS + 1) * CELL_SIZE,
            (2 * LOCAL_RADIUS + 1) * CELL_SIZE,
        )
        sensor_surf = pygame.Surface(
            ((2 * LOCAL_RADIUS + 1) * CELL_SIZE, (2 * LOCAL_RADIUS + 1) * CELL_SIZE),
            pygame.SRCALPHA,
        )
        sensor_surf.fill(C["sensor_box"])
        screen.blit(sensor_surf, (sensor_rect[0], sensor_rect[1]))
        pygame.draw.rect(screen, ROBOT_COLORS[focused_agent_id], sensor_rect, 1)

        # Draw all 4 AMRs
        for aid in env.agent_ids:
            col = ROBOT_COLORS[aid]
            rx, ry = env.positions[aid]
            rcx, rcy = cell_center(rx, ry)

            # Thicker border on focused robot
            if aid == focused_agent_id:
                pygame.draw.circle(screen, (255, 255, 255), (rcx, rcy), CELL_SIZE // 2, 2)

            pygame.draw.circle(screen, col, (rcx, rcy), CELL_SIZE // 3)
            lbl_r = font_sm.render(aid, True, (15, 15, 15))
            screen.blit(lbl_r, (rcx - lbl_r.get_width() // 2, rcy - lbl_r.get_height() // 2))

        # ── 4. Render Side Panel (HUD & Fleet Metrics) ───────────────
        px = GRID_SIZE * CELL_SIZE
        pygame.draw.rect(screen, C["panel_bg"], (px, 0, PANEL_W, SCREEN_H))
        pygame.draw.line(screen, C["divider"], (px, 0), (px, SCREEN_H), 2)

        py = 14

        def text(msg: str, font, color=C["text"], indent=16):
            nonlocal py
            surf = font.render(msg, True, color)
            screen.blit(surf, (px + indent, py))
            py += surf.get_height() + 3

        def sep(pad=6):
            nonlocal py
            py += pad

        # Header
        text("FLEET MARL TRAINING (4 AMRs)", font_title, C["accent"])
        text(f"Scenario: {layout_name}", font_bold, C["text"])
        sep(3)

        mode_lbl = " [FAST 10x]" if fast_mode else " [NORMAL]"
        if paused:
            mode_lbl = " [PAUSED]"
        text(f"Status: {status_text}", font_body, status_color)
        text(f"Engine: {mode_lbl}", font_sm, C["text_dim"])
        sep(4)

        # Fleet Progress
        text("FLEET OVERVIEW", font_bold, C["text"])
        text(f"Episode       : {episode:,}", font_body)
        text(f"Epsilon (Exp) : {agent.epsilon:.3f}", font_body)
        text(f"Fleet Steps   : {ep_steps} / {env.max_steps}", font_body)
        text(f"Peer Conflicts: {ep_peer_collisions}", font_body, C["warn"] if ep_peer_collisions > 0 else C["text_dim"])

        win_rate = (np.mean(recent_fleet_successes[-50:]) * 100.0) if recent_fleet_successes else 0.0
        text(f"Fleet Win Rate: {win_rate:.1f}% (All 4)", font_bold, ROBOT_COLORS["1"] if win_rate > 70 else C["accent"])
        sep(6)

        # Focused Robot Details
        text("FOCUSED ROBOT", font_bold, ROBOT_COLORS[focused_agent_id])
        text(f"Active Robot  : AMR {focused_agent_id} (Keys 1-4)", font_bold, ROBOT_COLORS[focused_agent_id])
        pos_str = f"{env.positions[focused_agent_id]} -> {env.goals[focused_agent_id]}"
        text(f"Route         : {pos_str}", font_body)
        text(f"Reward        : {ep_rewards[focused_agent_id]:.1f}", font_body)
        text(f"Collisions    : {ep_collisions[focused_agent_id]}", font_body)
        sep(6)

        # Live Q-Values for Focused Robot
        text(f"Q-VALUES (AMR {focused_agent_id})", font_bold, C["text"])
        sep(2)

        q_min, q_max = min(-10.0, float(np.min(q_values_np))), max(20.0, float(np.max(q_values_np)))
        q_range = max(1.0, q_max - q_min)

        for act_idx, act_name in enumerate(ACTION_LABELS):
            val = q_values_np[act_idx]
            is_best = act_idx == best_action

            lbl_col = ROBOT_COLORS[focused_agent_id] if is_best else C["text_dim"]
            surf_lbl = font_sm.render(f"{act_name:10s}: {val:6.1f}", True, lbl_col)
            screen.blit(surf_lbl, (px + 16, py))

            bar_x = px + 155
            bar_w = 140
            bar_h = 10

            pygame.draw.rect(screen, C["bar_bg"], (bar_x, py + 2, bar_w, bar_h))
            norm_val = max(0.0, min(1.0, (val - q_min) / q_range))
            fill_w = int(bar_w * norm_val)
            fill_col = ROBOT_COLORS[focused_agent_id] if is_best else C["bar_fill"]
            pygame.draw.rect(screen, fill_col, (bar_x, py + 2, fill_w, bar_h))

            py += 16

        sep(8)

        # Controls & Hotkeys
        text("CONTROLS", font_bold, C["text"])
        text("[1-4]    Focus AMR 1, 2, 3, or 4", font_sm, C["text_dim"])
        text("[SPACE]  Pause / Resume", font_sm, C["text_dim"])
        text("[F]      Toggle Fast 10x Speed", font_sm, C["text_dim"])
        text("[E]      Toggle Explore / Eval", font_sm, C["text_dim"])
        text("[R]      Re-roll Random Routes", font_sm, C["text_dim"])
        text("[N]      Next Scenario Layout", font_sm, C["text_dim"])
        text("[S]      Save Model Weights", font_sm, C["text_dim"])
        text("[ESC/Q]  Save & Exit", font_sm, C["text_dim"])

        pygame.display.flip()
        clock.tick(30 if not fast_mode else 120)

    # Clean Exit: save model
    agent.save(model_path)
    print(f"[Visualizer] Model weights saved to {model_path}")
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
