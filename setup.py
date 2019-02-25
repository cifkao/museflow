import setuptools

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
    install_requires=[
        'cached_property',
        'coloredlogs',
        'numpy',
        'pretty_midi',
        'pyyaml',
        'tensorflow',
    ],
)
