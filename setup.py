from setuptools import setup, find_packages

setup(
    name="geofuse",
    version="1.0.0",
    author="[Your Name]",
    description="A Multimodal Environmental Profiling Toolbox for Digital Health",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "geopandas",
        "rasterio",
        "shapely",
        "scipy",
        "tqdm",
        "earthengine-api",
        "geemap",
        "torch",
        "torchvision",
        "Pillow",
        "optuna",
        "scikit-learn",
        "skrebate",
        "streamlit",
        "folium",
        "streamlit-folium",
    ],
    entry_points={
        "console_scripts": [
            "geofuse=scripts.cli:main",
        ],
    },
)
