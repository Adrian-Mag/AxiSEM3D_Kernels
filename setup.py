from setuptools import setup, find_packages

# Load the contents of your README file
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='axikernels',
    version='0.1',
    description='A Python package for handling Axisem3D output',
    author='Marin Adrian Mag',
    author_email='marin.mag@stx.ox.ac.uk',
    packages=find_packages(exclude=['tests*']),
    package_data={
        'axikernels': ['examples/data/*'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
