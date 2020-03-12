import setuptools

setuptools.setup(
    name='gator',
    version='0.0',
    packages=['gator'],
    package_dir={'gator': 'src'},
    entry_points={'console_scripts': ['gator=gator.main:main']},
    python_requires='>=3.5',
    install_requires=['mpi4py>=3.0', 'numpy>=1.13'],
    description='GATOR',
    author='GATOR developers',
)
