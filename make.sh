all: build install

build:
       python3 setup.py build

install:
       python3 setup.py install --user
       cp sbb_column_classifier/sbb_column_classifier.py ~/bin/sbb_column_classifier
       chmod +x ~/bin/sbb_column_classifier
