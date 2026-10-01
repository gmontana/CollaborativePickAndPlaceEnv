from pathlib import Path

from setuptools import find_namespace_packages, setup

setup(
    name="collaborative_pick_and_place",
    version="0.1.0",
    description="Collaborative Pick and Place Environment",
    author="Giovanni Montana",
    long_description=Path(__file__).with_name("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/gmontana/CollaborativePickAndPlaceEnv",
    packages=find_namespace_packages(include=["macpp", "macpp.core*", "macpp.agents*"]),
    python_requires=">=3.10",
    install_requires=["gymnasium>=1.0,<2", "pygame>=2.5", "numpy>=1.23"],
    extras_require={
        "test": ["pytest>=7"],
        "agents": ["torch>=2", "wandb"],
        "video": ["imageio[ffmpeg]>=2.31"],
    },
    package_data={"macpp.core": ["icons/*.png"]},
    include_package_data=False,
)
