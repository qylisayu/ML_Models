Part I of the assignment are theoretical problems. See handout for more information.
Part II of the assignment investigates the performance of a simple neural network in classifying different letters using PyTorch


A2.py contains all the code to part 2 of the assignment. 

To run the code, make the required imports as displayed at the top of the file. The data path constant should be already assigned. If not, then ensure that the paths are pointing to the correct file path of the images in the "notMNIST_small" folder. Please add the folders of the data (A, B, C,...J) to this folder. The split data path constant should point to a new folder called "notMNIST_small_split". After running line 28 of the code (split folders.ratio()), this folder should be filled with the data that has been split into training, validation, and test sets.

Each model trained is defined and saved separately in the root folder. The output graphs are also saved in the folder. Running the A2.py file would re-train/test the models (so expect slightly different outputs each time). To generate all the graphs and models, please run the A2.py. Everything will be saved in the folder as outputs.