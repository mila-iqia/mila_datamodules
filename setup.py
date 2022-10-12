from setuptools import find_namespace_packages, setup

packages = find_namespace_packages(include=["mila_datamodules*"])
packages += find_namespace_packages(include=["hydra_plugins.*"])


setup(
    name="mila_datamodules",
    version="0.0.1",
    description="DataModules adapted for the Mila cluster.",
    author="Fabrice Normandin",
    author_email="fabrice.normandin@gmail.com",
    packages=packages,
    python_requires=">=3.7",
    # External packages as dependencies
    install_requires=[
        "pytorch-lightning",
        "torchvision",
        "lightning-bolts",
        "filelock",
    ],
    extras_require={
        "ffcv": "ffcv",
        "hydra": ["hydra-core", "hydra-zen"],
        "coco": "pycocotools",
        "test": ["pytest-xdist", "pytest-timeout"],
    },
    include_package_data=True,
)
