from setuptools import setup, find_packages

setup(
    name='collaborative_pick_and_place',
    version='0.0.1',
    description="Collaborative Pick and Place Environment",
    author="Giovanni Montana",
    packages=find_packages(),
    install_requires=[
        'gym',
        'pygame',
        'numpy'
    ],
    extras_require={"test": ["unittest"]},
    include_package_data=True,
)
