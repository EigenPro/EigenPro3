from setuptools import setup, find_packages
import eigenpro3

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='eigenpro3',
    version=eigenpro3.__version__,
    author='Amirhesam Abedsoltan, Parthe Pandit',
    author_email='aabedsoltan@ucsd.edu, parthepandit@ucsd.edu',
    description='Fast solver for learning general large scale kernel models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/EigenPro/EigenPro3',
    project_urls = {
        "Bug Tracker": "https://github.com/EigenPro/EigenPro3/issues"
    },
    license='Apache-2.0 license',
    packages=find_packages(),
    install_requires=['scipy'],
)
