from setuptools import setup

setup(
   name='jlab_datascience',
   version='0.1',
   description='JLab Data Science Toolkit',
   authors=['Malachi Schram', 'Kishansingh Rajput'],
   authors_email=['schram@jlab.org', 'kishan@jlab.org'],
   packages=['core'],
   install_requires=['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'scipy', 'tqdm', 'jupyter'],
)
