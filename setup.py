"""
This is a setup script for installing the parser locally on python path with
all the required dependencies. Used mainly for local testing.
"""
from setuptools import setup, find_packages


#===============================================================================
def main():
    # Start package setup
    setup(
        name="fhiaimsparser",
        version="0.1",
        description="NoMaD parser implementation for FHI-aims.",
        package_dir={'': 'parser/parser-fhi-aims'},
        packages=find_packages(),
        install_requires=[
            'numpy',
        ],
    )

# Run main function by default
if __name__ == "__main__":
    main()
