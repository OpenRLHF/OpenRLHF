from setuptools import find_packages, setup


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


def fetch_readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def fetch_version():
    with open('version.txt', 'r') as f:
        return f.read().strip()


setup(
    name='openllama2',
    version=fetch_version(),
    packages=find_packages(exclude=(
        'data',
        'docs',
        'examples',
    )),
    description='A LLaMA2 implementation',
    long_description=fetch_readme(),
    long_description_content_type='text/markdown',
    install_requires=fetch_requirements('requirements.txt'),
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Environment :: GPU :: NVIDIA CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Distributed Computing',
    ],
)
