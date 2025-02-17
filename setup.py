from setuptools import setup

setup(
    name="mlarchitectures",
    version="0.2.3",
    description="Shared ML Architectures",
    url="https://gitlab.au.dk/tech_ear-eeg/ml_architectures",
    author="Jesper Strøm",
    author_email="js@ece.au.dk",
    packages=[
        "ml_architectures",
        "ml_architectures.seqsleepnet",
        "ml_architectures.lseqsleepnet",
        "ml_architectures.usleep",
        "ml_architectures.common",
    ],
    #install_requires=["torch", "numpy"],
)
