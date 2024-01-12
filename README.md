# AMLS_23-24_SN23187901

To run the main.py file, add the npz format datasets into dataset folder first

main.py will load the datasets first and do data pre-processing for task A. Then find the best kernel and C value combination for SVM model using validation set. Use the optimized model to predict test set.
Then do data pre-processing for task B. Use Random Forest Model first and then CNN model.

# Role of each file

Task_A.py file in folder A contains SVM and CNN model for task A

Task_B.py file infolder B contains the Random Forest model and CNN model for task B

the figures and plots used to analyze the model in the report are both saved in the corresponding folder 

# Packages required to run the code

numpy
matplotlib
seaborn
scikit-learn
medmnist
tensorflow
random
