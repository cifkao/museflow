import setuptools

setuptools.setup(
    name="museflow",
    author="Ondřej Cífka",
    description="A music sequence learning framework",
    packages=['museflow'],
    install_requires=['tensorflow', 'pretty_midi', 'cached_property']
)
