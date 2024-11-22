from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()
VERSION = '0.0.1'
NAME = 'dkps'
setup(
    name=NAME,
    packages=find_packages(exclude=['tests', 'misc', 'asset']),
    version=VERSION,
    description='Compare foundation models.',
    url='https://github.com/hhelm10/dkps',
    keywords=['generative models', 'embeddings', 'populations of models'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Hayden Helm',
    author_email='hayden@helivan.io',
    install_requires=[
        "graspologic",
    ],
)
