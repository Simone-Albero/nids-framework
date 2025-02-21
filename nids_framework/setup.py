from setuptools import setup, find_packages

setup(
    name="nids_framework",
    version="0.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "torchvision",
        "torchaudio",
        "tqdm",
        "psutil",
        "matplotlib",
        "scikit-learn",
        "rich",
    ],
    author="Simone Albero",
    author_email="simone.albero@uniroma3.it",
    description="A Python framework for Network Intrusion Detection Systems (NIDS).",
    long_description_content_type="text/markdown",
    url="https://github.com/Simone-Albero/nids-framework.git",
    python_requires=">=3.6",
)
