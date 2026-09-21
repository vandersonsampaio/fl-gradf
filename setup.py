from setuptools import find_packages, setup

with open("requiriments.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="fedmd",
    version="0.1.0",
    description="FedMD — Federated Learning Security Framework (GRADF)",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=requirements,
    python_requires=">=3.10",
)
