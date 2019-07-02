import setuptools
import sys


gpu = True
if '--nogpu' in sys.argv:
  gpu = False
  sys.argv.remove('--nogpu')

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
        'numpy',
        'pretty_midi',
        'pyyaml',
        'tensorflow-gpu<2.0' if gpu else 'tensorflow<2.0',
    ],
)
