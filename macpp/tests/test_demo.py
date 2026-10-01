import importlib.util
from pathlib import Path


def test_readme_demo_completes_both_deliveries():
    path = Path(__file__).resolve().parents[2] / "examples" / "record_demo.py"
    spec = importlib.util.spec_from_file_location("record_demo", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    env = demo.demo_env()
    total_reward = 0
    try:
        for step, (actions, _) in enumerate(demo.DEMO_STEPS, start=1):
            obs, reward, terminated, truncated, _ = env.step(actions)
            assert env.observation_space.contains(obs)
            assert not truncated
            assert terminated == (step == len(demo.DEMO_STEPS))
            total_reward += reward
        assert total_reward == 146
        assert {obj.position for obj in env.objects} == set(env.goals)
        assert all(obj.carrying_agent is None for obj in env.objects)
    finally:
        env.close()
