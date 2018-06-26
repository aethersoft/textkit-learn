from setuptools import setup, find_packages

setup(name='tklearn',
      version='0.1',
      description='A Natural Language Learning Toolkit',
      url='',
      author='Yasas Senarath',
      author_email='wayasas@gmail.com',
      license='MIT',
      packages=find_packages(),
      package_dir={'tklearn': 'tklearn'},
      zip_safe=False,
      install_requires=['pandas', 'numpy', 'scipy', 'scikit-learn', 'nltk', 'keras', 'beautifulsoup4', 'texttable',
                        'gensim', 'hyperopt', 'jnius'])
