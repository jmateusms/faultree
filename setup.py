from setuptools import setup, find_packages

setup(
    name='faultree',
    version='0.1.0',
    description='Fault tree analysis library',
    author='João Mateus Santana',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
