import setuptools


setuptools.setup(
  name="scaled_cooking",
  version='1.0',
  install_requires=["google-cloud-storage", "apache-beam", "javalang", "numpy", "pickle", "json", "data_prep"],
  packages=setuptools.find_packages(),
)
