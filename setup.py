import setuptools
import sys

version = {}
with open('museflow/version.py') as f:
    exec(f.read(), version)

setuptools.setup(
    name="museflow",
    version=version['__version__'],
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
        'note-seq',
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
