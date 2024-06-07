from setuptools import setup

with open("requirements.txt", "r") as file:
    packages = file.read()

setup(
    name="fbpinns",
    packages=["fbpinns"],
    version="1.0.0",
    description="Packaging for PyTorch FBPINNs",
    author="Bejamin Mosely",
    install_requires=packages,
)
