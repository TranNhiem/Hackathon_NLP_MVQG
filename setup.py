from setuptools import setup, find_packages
from os.path import abspath, join, dirname


# read the contents of your README file
this_directory = abspath(dirname(__file__))
with open(join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='NLP Hackaphone project',
    version='0.0.1',
    description='NLP Hackaphone project for MVQG task chatbot',
    author='Group1',
    #long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/TranNhiem/Hackathon_NLP_MVQG',
    packages=find_packages(),
)
