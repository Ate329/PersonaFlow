from setuptools import setup, find_packages

# Core requirements
REQUIREMENTS = [
    "torch",
    "transformers>=4.45.0",
    "accelerate",
]

setup(
    name="personaflow",
    version="0.1.1",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    author="Zhiyong (Justin) He",
    author_email="justin.he814@gmail.com",
    description="A lightweight Python library for managing dynamic multi-persona interactions with LLMs.",
    long_description=open("README.md").read(),
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
