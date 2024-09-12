from setuptools import setup, find_packages

setup(
    name='nids_framework',
    version='0.0.0',
    package_dir={"": "src"}, 
    packages=find_packages(where="src"),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'torchvision',
        'torchaudio',
        'tqdm',
        'psutil',
        'rich',
        'matplotlib', 
        'scikit-learn'
    ],
    author='Simone Albero',
    author_email='sim.albero@stud.uniroma3.it',
    description='a Python framework for Network Intrusion Detection Systems (NIDS)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Simone-Albero/NIDS-Framework',
    python_requires='>=3.6',
)
