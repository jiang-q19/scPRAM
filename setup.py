"""Package metadata; the version is shared with the runtime package."""

from pathlib import Path

from setuptools import find_packages, setup

root = Path(__file__).parent
version = {}
exec((root / "scpram" / "version.py").read_text(), version)
setup(
    name="scpram",
    version=version["__version__"],
    description="Single-cell perturbation response prediction with attention and optimal transport",
    long_description=(root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/jiang-q19/scPRAM",
    author="Qun Jiang",
    packages=find_packages(include=["scpram", "scpram.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23,<3",
        "scipy>=1.9,<2",
        "pandas>=1.5,<3",
        "anndata>=0.9,<0.13",
        "scanpy>=1.9.3,<2",
        "scikit-learn>=1.2,<2",
        "POT>=0.9.3,<0.10",
        "torch>=2.6,<3",
        "tqdm>=4.64",
    ],
    extras_require={
        "plot": ["adjustText>=0.8,<2", "matplotlib>=3.6,<4", "seaborn>=0.12,<1"],
        "notebook": ["jupyter>=1,<2"],
        "dev": ["pytest>=7,<10", "build>=1,<2", "ruff>=0.11,<1"],
    },
)
