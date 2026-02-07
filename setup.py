"""
DeePAW: Deep Learning for PAW Charge Density Prediction
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    init_file = os.path.join(os.path.dirname(__file__), "deepaw", "__init__.py")
    with open(init_file, "r", encoding="utf-8") as f:
        content = f.read()
        version_match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="deepaw",
    version=get_version(),
    author="DeePAW Team",
    author_email="thsu0407@gmail.com",
    description="Deep Learning for PAW Charge Density Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Suth/DeePAW",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "e3nn>=0.5.0",
        "ase>=3.22.0",
        "pymatgen>=2023.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "tqdm>=4.60.0",
        "pykan>=0.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "accelerate>=0.20.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "accelerate>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepaw-server=deepaw.server.cli:server_main",
            "deepaw-predict=deepaw.server.cli:predict_main",
            "deepaw-predict-chgcar=deepaw.scripts.predict_chgcar:main",
        ],
    },
    include_package_data=True,
    package_data={
        "deepaw": ["checkpoints/*.pth"],
    },
    zip_safe=False,
)

