import setuptools
import sys

version = {}
with open('museflow/version.py') as f:
    exec(f.read(), version)

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="museflow",
    version=version['__version__'],
    author="Ondřej Cífka",
    author_email='ondra@cifka.com',
    description="Music sequence learning toolkit",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/cifkao/museflow',
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
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research'
    ],
)
