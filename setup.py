import setuptools
import sys


setuptools.setup(
    name="museflow",
    author="Ondřej Cífka",
    description="Music sequence learning toolkit",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'museflow = museflow.main:main'
        ]
    },
    python_requires='>=3.6',
    install_requires=[
        'cached_property',
        'coloredlogs',
        'confugue',
        'lmdb',
        'numpy',
        'pretty_midi',
        'pyyaml',
        'magenta.music @ git+https://github.com/cifkao/magenta@magenta.music',
    ],
    extras_require={
        'gpu': [
            'tensorflow-gpu<2.0',
        ],
        'nogpu': [
            'tensorflow<2.0',
        ]
    }
)
