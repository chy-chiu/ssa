from setuptools import find_packages, setup


setup(
    name="ssa",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={"ssa": ["assets/*.yaml", "assets/*.txt", "agents/prompts/*.json"]},
)
