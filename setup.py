from setuptools import find_namespace_packages, setup

packages = find_namespace_packages(include=["mila_datamodules*"])
packages += find_namespace_packages(include=["hydra_plugins.*"])

extras_require = {
    "ffcv": "ffcv",
    "hydra": ["hydra-core", "hydra-zen"],
    "coco": "pycocotools",
    "test": ["pytest-xdist", "pytest-timeout"],
}
extras_require["all"] = sorted(
    set(
        sum(
            ([dep] if isinstance(dep, str) else dep for dep in extras_require.values()),
            [],
        )
    )
)
with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f.read().splitlines() if not line.strip().startswith("#")
    ]

setup(
    name="mila_datamodules",
    version="0.0.1",
    description="DataModules adapted for the Mila / DRAC SLURM clusters.",
    author="Fabrice Normandin",
    author_email="normandf@mila.quebec",
    packages=packages,
    python_requires=">=3.9",
    # External packages as dependencies
    install_requires=requirements,
    # install_requires=[
    #     "pytorch-lightning",
    #     "torchvision",
    #     "lightning-bolts",
    #     "filelock",
    #     "pydantic",  # used in the cluster utils to populate the SlurmEnvVariables object.
    #     "scipy",  # required for Caltech datasets.
    # ],
    extras_require=extras_require,
    include_package_data=True,
)
