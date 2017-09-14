from setuptools import setup
from setuptools import find_packages


__VERSION__ = "0.1"

setup(
    name="pytorch-sru",
    version=__VERSION__,
    description="Simple Recurrent Unit (SRU) in PyTorch",
    url="https://github.com/kaniblu/pytorch-sru",
    author="Kang Min Yoo",
    author_email="k@nib.lu",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3"
    ],
    keywords="pytorch cuda deep learning",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "cupy",
        "pynvrtc"
    ],
    package_data={
        "": ["*.cu"]
    }
)