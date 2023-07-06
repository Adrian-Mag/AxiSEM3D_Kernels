from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

install_requires = []
for requirement in requirements:
    if not requirement.startswith('#') and not requirement.startswith('_'):
        package_name = requirement.split('=')[0]
        install_requires.append(package_name)

setup(
    name='axisem3d_output',
    version='1.0.0',
    description='A Python package for handling Axisem3D output',
    author='Marin Adrian Mag',
    author_email='marin.mag@stx.ox.ac.uk',
    packages=find_packages(),
    install_requires=install_requires,
)