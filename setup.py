from setuptools import setup, find_packages

setup(
    name='collaborative_pick_and_place',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'gym',
        'pygame',
        'numpy'
    ],
)
