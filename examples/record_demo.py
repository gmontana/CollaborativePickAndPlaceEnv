"""Render a verified, successful episode as a GIF and an MP4 for the README.

Run from the repository root: python examples/record_demo.py
Requires: pip install -e '.[video]'
"""

from pathlib import Path

import pygame

from macpp.core.environment import MACPPEnv

INITIAL_STATE = {
    "agents": {
        "agent_0": {"position": (0, 0), "picker": True, "carrying_object": None},
        "agent_1": {"position": (2, 0), "picker": False, "carrying_object": None},
    },
    "objects": [{"id": 0, "position": (1, 0)}, {"id": 1, "position": (0, 2)}],
    "goals": [(2, 1), (2, 2)],
}
# Integer actions follow Action: UP, DOWN, LEFT, RIGHT, PASS, WAIT.
DEMO_STEPS = [
    ([3, 5], "Picker collects the first box"),
    ([4, 4], "Both agents PASS to transfer it"),
    ([2, 1], "Dropper delivers to the first goal"),
    ([1, 2], "Agents move toward the second box"),
    ([1, 1], "Picker collects the second box"),
    ([4, 4], "Both agents PASS again"),
    ([5, 3], "Success! Both boxes delivered"),
]


def demo_env():
    return MACPPEnv(
        (3, 3), 2, 1, 2, initial_state=INITIAL_STATE, render_mode="rgb_array", cell_size=128
    )


def captioned_frame(env, step, caption, reward_total):
    """Add status text around the actual environment render, without a window."""
    grid = env.render()
    surface = pygame.Surface((480, 512))
    surface.fill((246, 248, 252))
    title_font = pygame.font.Font(None, 29)
    label_font = pygame.font.Font(None, 23)
    small_font = pygame.font.Font(None, 21)
    title = title_font.render("Collaborative Pick and Place", True, (24, 35, 54))
    surface.blit(title, title.get_rect(center=(240, 24)))
    subtitle = small_font.render("Picker + Dropper  |  2 boxes  |  7 steps", True, (68, 81, 103))
    surface.blit(subtitle, subtitle.get_rect(center=(240, 48)))
    surface.blit(pygame.surfarray.make_surface(grid.transpose(1, 0, 2)), (48, 64))
    text = label_font.render(caption, True, (24, 35, 54))
    surface.blit(text, text.get_rect(center=(240, 470)))
    delivered = sum(obj.position in env.goals and obj.carrying_agent is None for obj in env.objects)
    status = small_font.render(
        f"Step {step}/7    Delivered {delivered}/2    Team return {reward_total:+d}",
        True,
        (68, 81, 103),
    )
    surface.blit(status, status.get_rect(center=(240, 495)))
    return pygame.surfarray.array3d(surface).transpose(1, 0, 2).copy()


def record_demo(output_dir=None):
    import imageio.v2 as imageio

    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[1] / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    pygame.font.init()
    env = demo_env()
    frames = []
    durations = []
    total_reward = 0
    try:
        frames.append(captioned_frame(env, 0, "Two roles working together", total_reward))
        durations.append(1400)
        for step, (actions, caption) in enumerate(DEMO_STEPS, start=1):
            _, reward, terminated, truncated, _ = env.step(actions)
            total_reward += reward
            assert not truncated and terminated == (step == len(DEMO_STEPS))
            frames.append(captioned_frame(env, step, caption, total_reward))
            durations.append(2200 if terminated else 1100)
        assert all(obj.carrying_agent is None and obj.position in env.goals for obj in env.objects)
        imageio.mimsave(output_dir / "successful-episode.gif", frames, duration=durations, loop=0)
        # Repeat each rendered frame at 10 fps to match the GIF timing.
        with imageio.get_writer(output_dir / "successful-episode.mp4", fps=10) as writer:
            for frame, duration in zip(frames, durations):
                for _ in range(duration // 100):
                    writer.append_data(frame)
        print(f"Saved successful-episode.gif and .mp4 to {output_dir}; team return {total_reward}.")
    finally:
        env.close()
        pygame.font.quit()


if __name__ == "__main__":
    record_demo()
