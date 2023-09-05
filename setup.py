from setuptools import setup

setup(
    name='mlarchitectures',
    version='0.1.1',    
    description='Shared ML Architectures',
    url='https://gitlab.au.dk/tech_ear-eeg/ml_architectures',
    author='Jesper Strøm',
    author_email='js@ece.au.dk',
    packages=['architectures'],
    install_requires=['torch',
                      'numpy']
)