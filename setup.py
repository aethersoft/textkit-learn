import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as fh:
    requirements = fh.read().strip().split('\n')

setuptools.setup(
    name="textkit-learn",
    version="0.0.6",
    author="Yasas Senarath",
    description="Helps computers to understand human languages.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ysenarath/textkit-learn",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
