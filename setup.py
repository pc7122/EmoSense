import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().split("\n")

setuptools.setup(
    name='emosense',
    version='0.0.1',
    author='<NAME>',
    author_email='<EMAIL>',
    description='Multimodal emotion recognition',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/pc7122/EmoSense',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.10.12',
    install_requires=requirements,
)
