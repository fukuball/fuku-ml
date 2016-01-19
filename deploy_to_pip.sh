#!/bin/sh

echo "deplody to pip..."

echo "delete build..."
rm -rf build
rm -rf dist
rm -rf FukuML.egg-info

echo "prepare for upload..."
python setup.py sdist
python setup.py bdist_wheel --universal

echo "upload..."
twine upload dist/*

echo "deplody to pip complete."
