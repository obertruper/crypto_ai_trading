"""
Setup script for Crypto AI Trading System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="crypto-ai-trading",
    version="1.0.0",
    author="AI Trading Team",
    author_email="team@aitrading.com",
    description="Полноценная система прогнозирования и торговли криптовалютными фьючерсами с использованием PatchTST",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crypto-ai-trading",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pytest-cov>=3.0.0",
        ],
        "viz": [
            "plotly>=5.14.0",
            "dash>=2.10.0",
            "streamlit>=1.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto-ai-trading=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "crypto_ai_trading": [
            "config/*.yaml",
            "config/*.yml",
        ],
    },
)