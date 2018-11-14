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
    install_requires=['tensorflow', 'pretty_midi', 'cached_property'],
)
