from setuptools import setup

setup(
    name             = 'kdtreecomp',
    version          = '0.1',
    description      = 'Package for creating a binning scheme based on kd-trees',
    url              = 'http://github.com/dmriser/kdtree-comparator',
    author           = 'David Riser',
    author_email     = 'dmriser@gmail.com',
    license          = 'MIT',
    packages         = ['kdtree'],
    install_requires = [
        'numpy',
        ],
    zip_safe = False
    )
