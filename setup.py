from setuptools import setup

from mol2vec import __version__

setup(name='mol2vec',
      version=__version__,
      description='Mol2vec - an unsupervised machine learning approach to learn vector representations of molecular substructures',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD 3-clause',
        'Programming Language :: Python :: 3.6',
        'Topic :: Cheminformatics :: Featurization',
      ],
      url='http://github.com/samoturk/mol2vec',
      author='Sabrina Jaeger, Simone Fulle, Samo Turk',
      author_email='sabrina.jaeger@t-online.de, fulle@bio.mx, samo.turk@gmail.com',
      license='BSD 3-clause',
      packages=['mol2vec'],
      install_requires=['numpy', 'gensim', 'tqdm', 'joblib', 'pandas', 'matplotlib', 'IPython', 'seaborn'],
      zip_safe=False,
      entry_points="""
      [console_scripts]
      mol2vec=mol2vec.app.mol2vec:run
      """
      )
