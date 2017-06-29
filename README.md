# RASTA-Project

## Report
The report is available at :  https://www.overleaf.com/9772511nsjrzzccdcqv

---

## Train on a simple example dataset

### install all python requirements (see python/requirements.txt)

   `pip3 install -r python/requirements.txt`

### to train decaf6 on wikipaintings_10 run: 

    `python3 python/run.py`
    
Run `python3 python/run.py -h` to see parameters
    
Default parameters are :
  * model_name = decaf6
  * batch_size = 32
  * epochs = 10
  * Horizontal_flip = False
  * train_path = data/wikipaintings_10/wikipaintings_train
  * val_path = data/wikipaintings_10/wikipaintings_val


## Evaluate a model

Once the model is trained, you can evaluate the model by running
    `python3 	python/evaluation.py --model_path=YOUR_MODEL_PATH`

The YOUR_MODEL_PATH references to the path of your .h5 saved model (see savings folder).

There is 2 types of evaluations :
  * Accuracy evaluation : The top-k accuracy of the model is calculated given a specified training folder
  * Prediction : The top-k prediction is calculated given a specified image

Run `python3 python/evaluation.py -h` to see parameters.

Default parameters are :
  * type = acc
  * k = 1
  * --isdecaf = False
  * --data_path = data/wikipainting_10/wikipainting_10_test
  * --model_path = None

--data_path argument can reference the test folder or the image file, depending on the type of evaluation.
--isdecaf is necessary for the moment, because the evaluation for a decaf model is not exactly the same, due to a bug that will be fixed in the near future.



