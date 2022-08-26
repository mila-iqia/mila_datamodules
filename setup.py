from setuptools import find_namespace_packages, find_packages, setup

setup(
    name="mila_datamodules",
    version="0.0.1",
    description="DataModules adapted for the Mila cluster.",
    author="Fabrice Normandin",
    author_email="fabrice.normandin@gmail.com",
    packages=find_packages(include=["mila_datamodules.*"]),
    python_requires=">=3.7",
    # External packages as dependencies
    install_requires=[
        "pytorch-lightning==1.6.0",
        "lightning-bolts==0.5.0",
    ],
    extras_require={"ffcv": "ffcv"},
)
