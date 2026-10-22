from pathlib import Path

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


def parse_requirements(path="requirements.txt"):
    """Read requirements.txt so `pip install -e .` pulls the same deps.

    Skips blank lines, comments, inline comments, and pip flags (e.g. -r/-f/--index-url).
    """
    requirements = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            requirements.append(line)
    return requirements


setup(
    name="happy",
    version="3",
    author="Nellaker Group",
    description="The Histology Analysis Pipeline.py (HAPPY)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nellaker-group/happy",
    packages=find_packages(include=["happy"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7.2",
    install_requires=parse_requirements(),
    entry_points={
        "console_scripts": [
            "cell_infer=cell_inference:main",
            "nuc_train=nuc_train:main",
            "cell_train=cell_train:main",
            "tissue_infer=tissue_inference:main",
            "tissue_train=tissue_train:main",
        ]
    },
)
