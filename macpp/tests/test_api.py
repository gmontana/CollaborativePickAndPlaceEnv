"""Exercise public constructors, Gymnasium wrappers and the interactive example."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import pygame

from macpp.core import MACPPEnv
from macpp.core.environment import MACPPEnv as EnvironmentClass
from macpp.core.environment import make_env


def test_public_imports_and_factory_use_gymnasium():
    assert MACPPEnv is EnvironmentClass
    env = make_env(3, 3, 2, 1, 1)()
    assert isinstance(env, MACPPEnv)
    assert isinstance(env, gym.Env)
    obs, info = env.reset(seed=42)
    assert info == {}
    assert obs["agent_0"]["self"]["carrying_object"] == -1
    assert env.observation_space.contains(obs)
    assert len(env.step([5, 5])) == 5
    env.close()


def test_time_limit_is_truncation_not_task_completion():
    env = gym.make("macpp-3x3-2a-1p-2o-v0", max_episode_steps=1)
    try:
        env.reset(seed=42)
        _, reward, terminated, truncated, _ = env.step([5, 5])
        assert reward == -2
        assert not terminated
        assert truncated
        assert not env.unwrapped.done
    finally:
        env.close()


def test_import_does_not_load_legacy_gym():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, macpp; from macpp.core import MACPPEnv; assert 'gym' not in sys.modules",
        ],
        check=True,
    )


def test_interactive_episode_uses_five_value_steps(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    path = Path(__file__).resolve().parents[2] / "interactive.py"
    spec = importlib.util.spec_from_file_location("interactive", path)
    interactive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interactive)
    state = {
        "agents": {
            "agent_0": {"position": (0, 2), "picker": True, "carrying_object": None},
            "agent_1": {"position": (1, 0), "picker": False, "carrying_object": 0},
        },
        "objects": [{"id": 0, "position": (1, 0)}],
        "goals": [(2, 0)],
    }
    env = MACPPEnv((3, 3), 2, 1, initial_state=state, render_mode="human", cell_size=20)
    events = iter(
        [
            [
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p),
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT),
            ]
        ]
    )
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    try:
        interactive.game_loop(env)
        assert env.done
        assert env.objects[0].position == (2, 0)
        assert env.objects[0].carrying_agent is None
        assert env.renderer is None
    finally:
        env.close()
