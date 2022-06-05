``polaris2``
----

``polaris2`` is for reconstructing spatio-angular orientations from lightfield data. This project will succeed polaris with an improved class design and no dependence on dipy, vtk7, or python 3.5.

Installation instructions:

```
git clone https://github.com/talonchandler/polaris2.git
cd polaris2
conda env create -f environment.yml
conda activate polaris2
```

Run the example:

```
cd examples
python test-lf-psf.py
```