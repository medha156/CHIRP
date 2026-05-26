"""
Minimal setup.py for editable installs (`pip install -e .`).

Canonical metadata lives in pyproject.toml; this file only exists so that
tools that don't yet understand PEP 660 (pure pyproject editable installs)
can still do `pip install -e .` without complaints.
"""
from setuptools import setup

setup()
