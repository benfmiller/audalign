from setuptools import setup, find_packages


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
PACKAGE_VERSION = "0.0.2"
SUMMARY = "Audalign: Audio Alignment in Python"
DESCRIPTION = "This package offers fingerprinting, Recognizing, and aligning tools."

REQUIREMENTS = parse_requirements("requirements.txt")

setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    packages=find_packages(),
    license="MIT",
    description=SUMMARY,
    long_description=DESCRIPTION,
    author="Ben Miller",
    author_email="ben.f.miller24@gmail.com",
    maintainer="Ben Miller",
    maintainer_email="ben.f.miller24@gmail.com",
    url="http://github.com/benfmiller/audalign",
    include_package_data=True,
    platforms=["Unix", "Windows"],
    install_requires=REQUIREMENTS,
    keywords="python, audio, fingerprinting, music, numpy, landmark",
)
