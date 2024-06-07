from setuptools import setup

with open("requirements.txt", "r") as file:
    packages = file.read()

setup(
    name="fbpinns",
    packages=["fbpinns"],
    version="1.0.0",
    description="Packaging for PyTorch FBPINNs",
    author="Bejamin Mosely, Benedict Grey",
    author_email="bpg23@ic.ac.uk",
    install_requires=packages,
)
