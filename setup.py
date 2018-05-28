from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

    setup(
        name='SLJassBotSchieber',
        version='0.1.0',
        description='Supervised Learning JassBot for Schieber',
        long_description=readme,
        author='Andi Eder',
        author_email='andi.eder@gmx.ch',
        url='https://github.com/andieder/pyschieberSLJassBot',
        license='MIT',
        packages=find_packages(exclude='tests'),
    )
