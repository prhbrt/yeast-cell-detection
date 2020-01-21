from setuptools import setup, find_packages


setup(
    name='yeastcells',
    version=__import__('yeastcells').__version__,

    description='Computer vision based yeast cell detection.',
    long_description='Yeast cell detection using a deep convolutional network to detect cells, and classic computer vision, DBSCAN clustering and machine learning to process this to a geneology. Current state allows detection of cells, clustering over time to determine which are the same cell, finding cell boudnary using seam carving and prunign false positives.',

    url='https://git.webhosting.rug.nl/Research-Innovation-Support/yeastcells/',

    author='Herbert Kruitbosch',
    author_email='H.T.Kruitbosch@rug.nl',
    license='Only for use within the University of Groningen and only with permission from the authors.',

    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: Only with authors permission',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='yeast cell detection, microscopy images, tif, tiff, seam carving, u-net, deep learning, image segmentation, computer vision, DBSCAN',
    packages=find_packages(exclude=[]),
    include_package_data=True,

    install_requires=[
        'Keras>=2.3.1',
        'tensorflow>=2.0.0',
        'scikit-image>=0.16.1',
        'scikit-learn>=0.21.3',
        'opencv-contrib-python>=4.1.1.26',
        'numpy>=1.17.2',
        'scipy>=1.3.1',
        'tqdm>=4.36.1',
    ],
    extras_require={
        'dev': [],
        'test': [],
    },
    zip_safe=False,
)
