import setuptools

setuptools.setup(
    name="museflow",
    author="Ondřej Cífka",
    description="A music sequence learning framework",
    packages=['museflow'],
    entry_points={
        'console_scripts': [
            'museflow = museflow.main:main'
        ]
    },
    install_requires=[
        'cached_property',
        'numpy',
        'pretty_midi',
        'pyyaml',
        'tensorflow',
    ],
)
