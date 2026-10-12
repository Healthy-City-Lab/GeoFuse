from setuptools import find_packages, setup

setup(
    name="geofuse",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={"geofuse.presets": ["builtin/*/*.json"]},
    entry_points={"console_scripts": ["geofuse = geofuse.cli:main"]},
)
