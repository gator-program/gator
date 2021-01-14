import setuptools

setuptools.setup(
    name='gator',
    version='0.1',
    packages=['gator'],
    package_dir={'gator': 'src'},
    entry_points={'console_scripts': ['gator=gator.main:main']},
    python_requires='>=3.6',
    install_requires=['mpi4py>=3.0',
                      'numpy>=1.14',
                      'adcc>=0.15.6',
                      'psutil'],
    tests_require=['pytest'],
    description='GATOR',
    author='GATOR developers',
)
