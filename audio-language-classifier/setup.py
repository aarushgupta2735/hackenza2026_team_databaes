from setuptools import setup, find_packages

setup(
    name='audio-language-classifier',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project to classify audio samples by language.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.24',
        'scipy>=1.10',
        'librosa>=0.10',
        'soundfile>=0.12',
        'torch>=2.0',
        'transformers>=4.30',
        'lightgbm>=4.0',
        'scikit-learn>=1.3',
        'joblib>=1.3',
        'pandas>=2.0',
        'matplotlib>=3.7',
        'seaborn>=0.12',
        'tqdm>=4.65',
        'datasets>=2.14',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)