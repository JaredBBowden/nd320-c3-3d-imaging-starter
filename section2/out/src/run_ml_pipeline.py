"""
This file contains code that will kick off training and testing processes
"""
import os
import json

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData
import numpy as np


class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"/home/workspace/src/data" 
        self.n_epochs = 5 
        self.learning_rate = 0.0002
        self.batch_size = 32 # TODO Let's bump this up a smidge (was 8)
        self.patch_size = 64
        self.test_results_dir = "/home/workspace/out/" # Unclear is this is the best place for this (update: arbitrary, but fine)

if __name__ == "__main__":
    # Get configuration

    # TASK: Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()

    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)


    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # TASK: create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    # <YOUR CODE GOES HERE>
      
    # FIXME typically we would use some randomization in making these splits, however I don't believe we 
    # have reason to suspect there is any particular pattern to this dataset
    
#     split['train'] = keys[:int(len(keys)*0.6)]
#     split['val'] = keys[int(len(keys)*0.6):int(len(keys)*0.9)]
#     split['test'] = keys[int(len(keys)*0.9):]
    
    # DONE change of heart: update with randomized sampling method
    key_lenth = len(keys)
    split['train'] = np.random.choice(keys, int(key_lenth*0.6), replace=False)
    split['val'] = np.random.choice(keys, int(key_lenth*0.2), replace=False)
    split['test'] = np.random.choice(keys, int(key_lenth*0.2), replace=False)
    
    # Set up and run experiment
    
    # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    # del dataset 

    # run training
    exp.run()

    # prep and run testing

    # TASK: Test method is not complete. Go to the method and complete it
    results_json = exp.run_test()

    results_json["config"] = vars(c)
    
    print("Write to: ", exp.out_dir, "results.json")
    
    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

