from setuptools import setup, find_packages

from mol2vec import __version__

setup(name='mol2vec',
      version=__version__,
      description='Mol2vec - an unsupervised machine learning approach to learn vector representations of molecular \
                    substructures',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD 3-clause',
        'Programming Language :: Python :: 3.6',
        'Topic :: Cheminformatics :: Featurization',
      ],
      url='http://github.com/samoturk/mol2vec',
      author='Samo Turk, Sabrina Jaeger, Simone Fulle',
      author_email='samo.turk@gmail.com, sabrina.jaeger@t-online.de, fulle@bio.mx',
      license='BSD 3-clause',
      packages=find_packages(),
      install_requires=['numpy', 'gensim', 'tqdm', 'joblib', 'pandas', 'matplotlib', 'IPython', 'seaborn'],
      zip_safe=False,
      entry_points="""
      [console_scripts]
      mol2vec=mol2vec.app.mol2vec:run
      """
      )
