# Generating data
- run_generate.py: to generate HDF5 data.

# Generating training/validation/test datasets
- split-train.py: to generate training/validation/test datsets.
    It splits the MANY HDF5 files (bio and xtal) into two 4 files: train_bio.hdf5 train_xtal.hdf5, valid_bio.hdf5 and valid_xtal.hdf5. These files are created in a way that the two sets (bio and xtal) are balanced in both the training and the validation dataset and to avoid the situation in which a slightly rotated version of the same structure is present in both training and validation sets.

# Training models
- arch_001_02.py: the CNN achitecture.
- run_learn_gpu.py: to train models.
- DeepRank code version: commit c4378dabdc5015055a344dcaa1d7c34b25d65abb

# Analysis
- write_data.py: Write out accuracy and loss values for each epoch
- plt_acc.py: Plot training and validation accuracy for each epoch
- plt_loss.py: Plot training and validation losses for each epoch
- confusion_matrix.R: Plot confusion matrix