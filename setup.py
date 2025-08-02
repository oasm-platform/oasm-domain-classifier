from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="domain-classifier",
    version="0.1.0",
    author="OASM Platform",
    author_email="",
    description="A domain classification tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oasm-platform/oasm-domain-classifier",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "huggingface-hub>=0.10.0",
        "requests>=2.25.0",
        "beautifulsoup4>=4.9.0",
        "urllib3>=1.26.0",
    ],
    entry_points={
        "console_scripts": [
            "domain-classifier=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
