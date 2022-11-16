from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="happy",
    version="0.0.1",
    author="Claudia Vanea",
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
    entry_points={
        "console_scripts": [
            "cell_infer=cell_inference:main",
            "nuc_train=nuc_train:main",
            "cell_train=cell_train:main",
        ]
    },
)
