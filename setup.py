from datetime import datetime
from setuptools import find_packages, setup

import os


_build_mode = os.getenv("OPENRLHF_BUILD_MODE", "")


def _is_nightly():
    return _build_mode.lower() == "nightly"


def _fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def _fetch_readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def _fetch_version():
    with open("version.txt", "r") as f:
        raw_version_number = f.read().strip()
        return (
            f'{raw_version_number}{datetime.today().strftime("b%Y%m%d.dev0")}' if _is_nightly() else raw_version_number
        )


def _fetch_package_name():
    return "openrlhf-nightly" if _is_nightly() else "openrlhf"


setup(
    name=_fetch_package_name(),
    version=_fetch_version(),
    packages=find_packages(
        exclude=(
            "data",
            "docs",
            "examples",
        )
    ),
    description="A Ray-based High-performance RLHF framework.",
    long_description=_fetch_readme(),
    long_description_content_type="text/markdown",
    install_requires=_fetch_requirements("requirements.txt"),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)
