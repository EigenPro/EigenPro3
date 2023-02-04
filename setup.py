import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='eigenpro',
    version='3.0.0',
    author='Amirhesam Abedsoltan, Parthe Pandit',
    author_email='aabedsoltan@ucsd.edu, parthepandit@ucsd.edu',
    description='Fast solver for Kernel Regression using GPUs with linear space and time complexity',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/EigenPro/EigenPro3',
    project_urls = {
        "Bug Tracker": "https://github.com/EigenPro/EigenPro3/issues"
    },
    license='Apache-2.0 license',
    packages=['eigenpro'],
    install_requires=[
        'scipy >= 1.9',
    ],
)