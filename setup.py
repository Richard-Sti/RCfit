from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    # Basic package information:
    name="rcfit",
    version="0.1",
    packages=find_packages(),

    # Package metadata:
    author="Richard Stiskalek",
    author_email="richard.stiskalek@protonmail.com",
    description="RC Fits..",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Richard-Sti/RCfit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "scipy",
        "jax",
        "jaxlib",
        "numpyro",
        "h5py",
        "tqdm",
        "matplotlib",
        "corner",
        "scienceplots",
        "quadax"
        ],
)
