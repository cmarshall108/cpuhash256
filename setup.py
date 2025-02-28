from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

extensions = [
    Extension(
        "cpuhash256.cpuhash256_wrapper",
        sources=["cpuhash256/cpuhash256_wrapper.pyx", "cpuhash256/cpuhash256.c"],
        include_dirs=["cpuhash256"],
    )
]

setup(
    name="cpuhash256",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    package_data={
        'cpuhash256': ['*.pxd', '*.h'],
    },
    install_requires=[
        'numpy',
    ],
    python_requires='>=3.6',
    author="Caleb Marshall",
    description="A very fast CPU-based hash function implementation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
