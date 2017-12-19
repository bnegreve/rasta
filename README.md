# RASTA

The Rasta project aims at recognising art style automatically using pixel data only (i.e. no meta-data). For more details about the methodology and the results, see the publication.

Adrian Lecoutre, Benjamin Negrevergne and Florian Yger. Rasta: Recognizing art style automatically in painting with deep learning. In Asian Conference on Machine Learning, 2017.

# How to use this code

## Setup Rasta Python environment with Pip

You can install  python requirements with

    pip3 install -r python/requirements.txt

See `python/requirements.txt` for the complete list of requirements.

If you have access to GPUs  we encourage you to use them, this will speedup both inference and training. To use GPU, install tensorflow-gpu in addition to previous packages.

    pip3 install tensorflow-gpu


## Download model files (mandatory)

    cd models
    wget www.lamsade.dauphine.fr/~bnegrevergne/webpage/software/rasta/rasta_models.tgz
    tar xzvf rasta_models.tgz
    cd ../

## Download extra data files (optional)

   If you want to download the full wikipaintings dataset (the one from WikiArt), execute the following commands. Warning: the file is about ~20GiB, we suggest that you first try with the small datasets provided in `data/wikipaintaings_small`.

    cd data
    wget www.lamsade.dauphine.fr/~bnegrevergne/webpage/software/rasta/wikipaintaings_full.tgz
    tar xzvf wikipaintings_full.tgz
    cd ../

## Predict the style of one image

Use

    python3 python/evaluation.py -t pred  --data_path=PATH_TO_IMAGE

Where `PATH_TO_IMAGE` points toward a valid jpeg image file.

See `python3 python/evaluation.py -h` for more details 

## Evaluate Rasta models on a large batch of images

You can evaluate the accuracy of the default Rasta model using:

    python3 python/evaluation.py

This will evaluate the accuracy on a the small test set available in `wikipaintings_small/wikipaintings_test` using the model in `models/default`.

You can evaluate the accuracy of other Rasta models, or using other datasets with:

    python3 python/evaluation.py --model_path=MODEL_PATH --data_path=DATA_PATH

where `MODEL_PATH` is the path to a .h5 model file, and  `DATA_PATH` is a path to a directory containing the test set. In the test set, there should be one sub-directory for each class containing all the images of this class. See `wikipaintings_small/wikipaintings_test` for an example. 


See `python3 python/evaluation.py -h` for more details about the options.

Note: At the moment `--isdecaf` is necessary if you want to evaluate models based on decaf. Hopefully, this will be fixed soon. 

# License

See LICENSE file.

# Authors

- Adrian Lecoutre
- Benjamin Negrevergne
- Florian Yger

# Contact author

Benjamin Negrevergne: firstname.lastname @ dauphine.fr


