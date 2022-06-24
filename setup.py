import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='HomoPy',  
     version='0.2',
     author="Nicolas Christ",
     author_email="nicolas.christ@kit.edu",
     description="Your solution for stiffness problems",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/Extraweich/HomoPy",
     packages=["HomoPy"],
     install_requires=["numpy","matplotlib","scipy"],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )