"""
Setup script for Memory SDK.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="memory-sdk",
    version="0.1.0",
    author="Memory SDK Team",
    description="Production-grade Long-Term and Short-Term Memory for AI Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/memory-sdk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "redis>=5.0.0",
        "pymilvus>=2.4.0",
        "openai>=1.0.0",
        "httpx>=0.25.0",
    ],
    extras_require={
        "langgraph": [
            "langgraph>=0.2.0",
            "langchain-core>=0.3.0",
            "langchain-openai>=0.2.0",
        ],
        "anthropic": [
            "anthropic>=0.30.0",
        ],
        "all": [
            "langgraph>=0.2.0",
            "langchain-core>=0.3.0",
            "langchain-openai>=0.2.0",
            "anthropic>=0.30.0",
        ],
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "black>=24.0.0",
            "ruff>=0.3.0",
        ],
    },
)

