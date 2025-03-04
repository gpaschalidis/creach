from setuptools import setup, find_packages

def setup_package():

    setup(
        name="creach", 
        version="0.1.0", 
        author="Georgios Paschalidis", 
        description="CReach model",
        url="https://github.com/gpaschalidis/creach",
        packages=find_packages(exclude=["docs", "scripts"]),  
        install_requires=[], 
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
       ],
        python_requires=">=3.9",  
    )

if __name__ == "__main__":
    setup_package()

