# RASTA-Project

## Report
The report is available at :  https://www.overleaf.com/9772511nsjrzzccdcqv

---
## Project

First of all, datasets should be downloaded by running : `???`

A test file can be ran with the command : `python ./scripts/test.py`

For the moment, it runs a training on AlexNet with 1 epochs and 1 step per epoch. This can be modified by directly editing the file.

## Train on a simple example dataset

install all python requirements (see python/requirements.txt)

    cat python/requirements.txt | xargs pip3 install

to train decaf6 on wikipaintings_10 run: 

    python3 python/run.py decaf6 2 3 4 5 False data/wikipainting_10/wikipainting_10_train/ data/wikipainting_10/wikipainting_10_val/

where:

    decaf6 is the type of the network (see python/run.py for more details)
    2 is the batch size
    3 is the number of epochs
    4 is the number of steps per epoch
    5 is the number of steps per validation epoch

