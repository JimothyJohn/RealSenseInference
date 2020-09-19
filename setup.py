import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="realsenseinference", # Replace with your own username
    version="0.0.1",
    author="Nick Armenta",
    author_email="nick@advin.io",
    description="Integrates Nvidia Jetson Inference with Intel RealSense functionality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JimothyJohn/RealSenseInference",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu",
    ],
    python_requires='>=3.6',
)