from setuptools import setup
import sys
import setuptools
import glob

__version__ = '0.0.3'

setup(
    name='symplectic_map',
    version=__version__,
    author='Carlo Emilio Montanari',
    author_email='carlidel95@gmail.com',
    url='https://github.com/carlidel/c_symplectic_map',
    description='A Numba implementation of a Symplectic Map with nice python bindings',
    packages=['symplectic_map'],
    install_requires=['numba', 'numpy', 'scipy', 'tqdm'],
    setup_requires=['numba', 'numpy', 'scipy', 'tqdm'],
    license='MIT',
)
