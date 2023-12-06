from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name="scpram",
    version="0.0.1",
    description="scPRAM accurately predicts single-cell gene expression perturbation response based on attention mechanism",
    long_description="With the rapid advancement of single-cell sequencing technology, we are gradually enabled to delve into the cellular re-sponses to various external perturbations at the gene expression level. However, obtaining perturbed samples in certain scenarios may be considerably challenging, and the substantial costs associated with sequencing also curtail the feasibil-ity of large-scale experimentation. A repertoire of methodologies has been employed for forecasting perturbative re-sponses in single-cell gene expression. However, existing methods only focus on predictions at the average level and do not capture the heterogeneity at the single-cell resolution effectively. Here we present scPRAM, a method for predicting Perturbation Responses in single-cell gene expression based on At-tention Mechanisms. Leveraging variational autoencoders and optimal transport, scPRAM aligns cell states before and after perturbation, followed by accurate prediction of gene expression responses to perturbations for unseen cell types through attention mechanisms. Experiments on multiple real perturbation datasets involving drug treatments and bacteri-al infections demonstrate that scPRAM attains heightened accuracy in perturbation prediction across cell types, species, and individuals, surpassing existing methodologies. Furthermore, scPRAM demonstrates outstanding capability in identi-fying differentially expressed genes under perturbation, capturing heterogeneity in perturbation responses across spe-cies, and maintaining stability in the presence of data noise and sample size variations.",
    license="MIT Licence",
    url="https://github.com/jiang-q19/scPRAM",
    author="Qun Jiang",
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
    keywords="single cell, perturbation, attention, optimal transport",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        'scanpy==1.9.3',
        'torch==2.1.0',
        'torchaudio==2.1.0',
        'torchvision==0.16.0',
        'adjusttext==0.7',
        'pot',
        'jupyter'
    ]
)