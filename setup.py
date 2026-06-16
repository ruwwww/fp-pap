"""
setup.py - install the project as a local package for clean imports
"""
from setuptools import setup, find_packages

setup(
    name="eas_timeseries",
    version="1.0.0",
    packages=find_packages(),
    description="EAS Time Series Forecasting Environment",
    python_requires=">=3.9",
)
