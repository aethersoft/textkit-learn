import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tklearn",  # Replace with your own username
    version="v0.0.3",
    author="Yasas Senarath",
    description="Helps computers to understand human languages.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ysenarath/textkit-learn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
