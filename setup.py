from setuptools import setup, find_packages

setup(
    name="vegetable-freshness-api",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask==2.3.3",
        "flask-cors==4.0.0",
        "scikit-learn==1.3.2",
        "joblib==1.3.2",
        "numpy==1.23.5",
        "scipy==1.10.1",
    ],
    python_requires=">=3.11,<3.12",
)