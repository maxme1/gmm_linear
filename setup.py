from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

setup(
    name='gmm_linear',
    packages=find_packages(include=('gmm_linear',)),
    install_requires=requirements,
)
