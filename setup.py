from setuptools import setup, find_packages

setup(
    name="faultree",
    version="0.2.1",
    description="Fault Tree Analysis using BDDs",
    packages=find_packages(),
    install_requires=[
        "dd>=0.5",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "openpyxl>=3.0.0",
    ],
    extras_require={
        "server": ["fastapi", "uvicorn"],
        "all": ["fastapi", "uvicorn"],
    },
    entry_points={
        "console_scripts": [
            "faultree=faultree.cli:main",
        ],
    },
)
