import setuptools

setuptools.setup(
    name='gator',
    version='0.1',
    packages=['gator'],
    package_dir={'gator': 'src'},
    entry_points={'console_scripts': ['gator=gator.main:main']},
    python_requires='>=3.6',
    # install_requires=['mpi4py>=3.0',
    #                   'numpy>=1.14'],
    description='GATOR',
    author='GATOR developers',
)
