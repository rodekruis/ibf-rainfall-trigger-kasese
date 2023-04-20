"""
Setup.py file.
Install once-off with:  "pip install ."
For development:        "pip install -e .[dev]"
"""
import setuptools


with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

PROJECT_NAME = "pipeline"

setuptools.setup(
    name=PROJECT_NAME,
    version="0.1",
    author="Phuoc Phung",
    author_email="pphung@redcross.nl",
    description="App for heavy rainfall forecast",
    package_dir={"": "lib"},
    packages=setuptools.find_packages(where="lib"),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            f"run-pipeline = {PROJECT_NAME}.runPipeline:main",
        ]
    }
)