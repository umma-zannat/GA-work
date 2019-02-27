import os
from setuptools import setup, find_packages
from Cython.Build import cythonize

os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

setup(name='jami',
      description=('Python toolkit for data interpolation and visualisation'),
      version='0.0.0',
      author='Umma Jamila Zannat',
      author_email='umma.zannat@ga.gov.au',
      license='MIT',
      packages=find_packages(),
      python_requires='>=2.7, >=3.5',
      ext_modules=cythonize('jami/ndlinear/core.pyx'))
