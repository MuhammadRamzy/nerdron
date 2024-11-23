from setuptools import setup, find_packages

setup(
    name="nerdron",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "cupy-cuda11x>=11.0.0;platform_system!='Darwin'",
    ],
    author="Ramzy",
    author_email="mhdramzy777@gmail.com",
    description="A neural network library for fun",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MuhammadRamzy/nerdron",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)