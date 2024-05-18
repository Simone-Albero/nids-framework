from setuptools import setup, find_packages

setup(
    name='NIDS_Framework',
    version='0.0.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        # dependencies here
        'numpy',
        'pandas',
    ],
    author='Simone Albero',
    author_email='sim.albero@stud.uniroma3.it',
    description='a Python framework for Network Intrusion Detection Systems (NIDS)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Simone-Albero/NIDS-Framework',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
