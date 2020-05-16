# About
This repository describes our system for the task of Metaphor Detection.

## Information about files
* ```data_preparation.py``` is used for constructing datasets in the format of https://github.com/RuiMao1988/Sequential-Metaphor-Identification/tree/master/data which are prepared by https://github.com/gao-g/metaphor-in-context.
* ```model.py``` contains all the model classes.
* ```util.py``` contains all the helper functions.
* ```main_toefl.py``` contains the code for loading and running experiments on the [TOEFL](https://catalog.ldc.upenn.edu/LDC2014T06) dataset.
* ```main_vua.py``` contains the code for loading and running experiments on the [VUA](http://www.vismet.org/metcor/documentation/home.html) dataset.

## Environment
* The environment used is python 3.6 with pytorch 1.4 with standard libraries - allennlp, sklearn, numpy, pandas, matplotlib, nltk, tqdm etc.

## Data
* Run ```python util.py``` to make required directories.
* Download GloVe embeddings from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip), unzip them and place the text file in **./data/** folder.
* Download VUA data from [here](https://github.com/EducationalTestingService/metaphor/tree/master/VUA-shared-task) and prepare the following files - ```vuamc_corpus_train.csv```, ```vuamc_corpus_test.csv```, ```all_pos_test_tokens.csv``` and ```verb_test_tokens.csv``` and place all of these in **./data/vua/** folder.
* For downloading TOEFL dataset, you need to fill an agreement [here](https://github.com/EducationalTestingService/metaphor/blob/master/TOEFL-release/metaphor-shared-task-license-agreement.docx). Next, rename the **essays/** folder of training partition as **train_essays/** and place it in **./data/toefl/** folder, similarly rename **essays/** folder from test partition as **test_essays/** and place it in **./data/toefl/** folder. Also, place ```all_pos_test_tokens.csv``` and ```verb_test_tokens.csv``` in **./data/toefl/** folder.

## Use
* Run ```python data_preparation.py [option]```, where option *vua* creates all files (including ELMo vectors) for the VUA dataset and *toefl* for the TOEFL dataset. This script also splits the training dataset into train and validation sub parts. Note it takes time to compute the ELMo vectors.
* Run ```python main_xyz.py``` to run the experiments on the respective dataset. It will store the produced graphs in **./graphs/xyz/** folder. It also produces the test predictions which are stored as ```xyz_all_pos_pred.csv``` and ```xyz_verb_pred.csv``` in the **./predictions/** folder.

## Note
* The outputs here are expected to match the results reported in paper for the single run case.
* For ensembling, the code is not provided, one can run different models by varying hyperparameters of the model (as mentioned in paper) and aggregate by majority voting.