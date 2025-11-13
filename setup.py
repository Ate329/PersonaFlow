from setuptools import setup, find_packages  # type: ignore
from pathlib import Path

# Core requirements
REQUIREMENTS = [
    "torch",
    "transformers>=4.45.0",
    "accelerate",
    "typing-extensions",
]

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="personaflow",
    version="0.1.2",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    author="Zhiyong (Justin) He",
    author_email="justin.he814@gmail.com",
    description="A lightweight Python library for managing dynamic multi-persona interactions with LLMs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ate329/PersonaFlow",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.12",
)
