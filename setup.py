from setuptools import setup
import os

if not os.path.isdir("temp"):
    os.system("mkdir temp")
if not os.path.isdir("temp/2019-scalingattack"):
    os.system("git clone https://github.com/EQuiw/2019-scalingattack.git ./temp/2019-scalingattack/")

os.system("conda create --name image-scaling python=3.8")
os.system("conda activate image-scaling")

os.system("pip install -r requirements.txt")
os.system("python temp/2019-scalingattack/scaleatt/setup.py build_ext --inplace")

setup(
    name='imageScaling-attack-defense',
   version='1.0',
   description='Module attack defense',
   author='Erikson Aguiar',
   author_email='erjulioaguiar@usp.br',
)