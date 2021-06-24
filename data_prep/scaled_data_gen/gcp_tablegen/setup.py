import subprocess
import setuptools

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

setuptools.setup(
  name="dbwalk_setup",
  install_requires=["google-cloud-storage", "apache-beam"],
  packages=setuptools.find_packages(),
)