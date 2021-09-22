import pathlib

from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()


def parse_requirements(requirements):
    # load from requirements.txt
    with open(requirements) as f:
        lines = [l for l in f]
        # remove spaces
        stripped = list(map((lambda x: x.strip()), lines))
        # remove comments
        nocomments = list(filter((lambda x: not x.startswith("#")), stripped))
        # remove empty lines
        reqs = list(filter((lambda x: x), nocomments))
        return reqs


PACKAGE_NAME = "audalign"
PACKAGE_VERSION = "0.7.2"
SUMMARY = "Audio Alignment and Recognition in Python"

REQUIREMENTS = parse_requirements("requirements.txt")

setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    packages=["audalign"],
    license="MIT",
    description=SUMMARY,
    long_description=README,
    long_description_content_type="text/markdown",
    author="Ben Miller",
    author_email="benfmiller132@gmail.com",
    maintainer="Ben Miller",
    maintainer_email="benfmiller132@gmail.com",
    url="http://github.com/benfmiller/audalign",
    include_package_data=True,
    platforms=["Unix", "Windows"],
    install_requires=REQUIREMENTS,
    keywords="python, audio, align, alignment, fingerprinting, music",
)
