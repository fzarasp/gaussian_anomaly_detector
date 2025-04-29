from setuptools import setup, find_packages

setup(
    name='gaussian_anomaly_detector',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib'
    ],
    author='Fatemeh Zahra Safaeipour',
    author_email='fzarasp@gmail.com',
    description='Lightweight Gaussian-based anomaly detection for binary classification tasks.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/fzarasp/gaussian_anomaly_detector',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
    ],
)
