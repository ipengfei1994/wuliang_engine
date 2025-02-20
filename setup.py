from setuptools import setup, find_packages

setup(
    name="wuliang_engine",
    version="0.1.0",
    packages=find_packages(include=['engine', 'engine.*']),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ],
)