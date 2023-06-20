from distutils.core import setup
from setuptools import find_packages

setup(
    name='diffusion-models',
    version='0.0.1',
    py_modules=["diffusion_models"],
    license='MIT License',
    install_requires=[
            'torch>=1.12.1',
            'wandb',
            'absl-py',
            'ml-collections',
            'common-utils @ git+ssh://git@github.com/saeidnp/common-utils.git@master#egg=common_utils',
            'synthetic-datasets @ git+ssh://git@github.com/saeidnp/synthetic-datasets.git@master#egg=synthetic_datasets',
        ],
)