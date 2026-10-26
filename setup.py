from setuptools import setup, find_packages

setup(
    name="nasapowercouncil",
    version="2.1.0",
    description="Weather-based commodity futures trading research system",
    author="Alexander Robbins",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "yfinance>=0.2.28",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "report": ["quantstats>=0.0.62"],
        "xgboost": ["xgboost>=2.0"],
        "dev": ["pytest>=7.4.0"],
    },
    python_requires=">=3.9",
)
