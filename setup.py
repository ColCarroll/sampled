from codecs import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as buff:
    long_description = buff.read()

setup(
    name='sampled',
    version='0.1.1',
    description='Decorator for reusable models in PyMC3',
    long_description=long_description,
    author='Colin Carroll',
    author_email='colcarroll@gmail.com',
    url='https://github.com/ColCarroll/sampled',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(exclude=['test']),
    install_requires=[
        'pymc3>=3.0',
    ],
    include_package_data=True,
)
