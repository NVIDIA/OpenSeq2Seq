import setuptools
import numpy as np

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="open_seq2seq",
    version="0.0.1",
    author="voicezen",
    author_email="all@voicezen.ai",
    description="Python repo for components and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/voicezen/jivaka",
    packages=setuptools.find_packages(),
    include_dirs=[np.get_include()],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
