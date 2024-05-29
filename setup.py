from setuptools import setup, find_packages



with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='lit_ecology_classifier',
    version='0.1',
    description='Image Classifier optimised for ecology use-cases',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'torch',
        'torchvision',
        'torchaudio ',
        'lightning',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'lit_ecology_classifier=lit_ecology_classifier.main:main',
        ],
    },
    author='Benno Kaech',
    author_email='your.email@example.com',
    url='https://github.com/kaechb/lit_ecology_classifier',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
