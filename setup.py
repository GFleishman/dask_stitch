import setuptools

setuptools.setup(
    name="dask_stitch",
    version="0.1.1",
    author="Greg M. Fleishman",
    author_email="greg.nli10me@gmail.com",
    description="Linear blend stitching for map_overlap calls in dask",
    url="https://github.com/GFleishman/dask_stitch",
    license="MIT",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'dask',
        'dask[array]',
    ]
)

