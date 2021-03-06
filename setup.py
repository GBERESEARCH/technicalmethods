import pathlib
import runpy
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

version_meta = runpy.run_path("./version.py")
VERSION = version_meta["__version__"]
PACKAGE_NAME = 'technicalmethods'
AUTHOR = 'GBERESEARCH'
AUTHOR_EMAIL = 'gberesearch@gmail.com'
URL = 'https://github.com/GBERESEARCH/technicalmethods'

LICENSE = 'MIT'
DESCRIPTION = 'Technical Analysis tools'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=parse_requirements("requirements.txt"),
      packages=find_packages()
      )
