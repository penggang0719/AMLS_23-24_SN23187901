import numpy as np
import A.Task_A as A
import B.Task_B as B

#download the data if need, then save the npz file to dataset folder
# Pn = A.download_Pn_data()
# Pa = B.download_Pa_data()

# load and split the data from dataset folder use the load_data function
Pn_path = 'AMLS_23-24_SN23187901/dataset/pneumoniamnist.npz'
Pn, Pn_train, Pn_val, Pn_test, Pn_train_labels, Pn_val_labels, Pn_test_labels = A.load_data(Pn_path)
Pa_path = 'AMLS_23-24_SN23187901/dataset/pathmnist.npz'
Pa, Pa_train, Pa_val, Pa_test, Pa_train_labels, Pa_val_labels, Pa_test_labels = B.load_data(Pa_path)

# Count the number of normal and pneumonia in each dataset
train_normal, train_pneumonia = A.count_pneumonia(Pn_train_labels)
val_normal, val_pneumonia = A.count_pneumonia(Pn_val_labels)
test_normal, test_pneumonia = A.count_pneumonia(Pn_test_labels)

#display the distribution of normal and pneumonia in train, validation and testing dataset
A.distr_plot(train_normal, train_pneumonia,val_normal,val_pneumonia,test_normal,test_pneumonia,'distribution')

#Implement SVM model to validation dataset, find the best kernel and C value
Pn_train_flat = A.SVM_flatten(Pn_train)
Pn_val_flat = A.SVM_flatten(Pn_val)
Pn_test_flat = A.SVM_flatten(Pn_test)

linear_accuracies = []
poly_accuracies = []
rbf_accuracies = []

#set C from 0.1, 0.2 to 1, 1, 2 to 10 and 10, 20 to 100
C_values = np.concatenate((np.arange(0.1, 1.1, 0.1), np.arange(1, 11, 1),np.arange(10, 110, 10)))

for C in C_values:
    y_pred, poly_accuracy = A.SVM_model(Pn_train_flat, Pn_train_labels, Pn_val_flat, Pn_val_labels, 'poly', C)
    poly_accuracies.append(poly_accuracy)

for C in C_values:
    y_pred, linear_accuracy = A.SVM_model(Pn_train_flat, Pn_train_labels, Pn_val_flat, Pn_val_labels, 'linear', C)
    linear_accuracies.append(linear_accuracy)

for C in C_values:
    y_pred, rbf_accuracy = A.SVM_model(Pn_train_flat, Pn_train_labels, Pn_val_flat, Pn_val_labels, 'rbf', C)
    rbf_accuracies.append(rbf_accuracy)

#obtain the accuracy and plot the different kernel's accuracy vs C values
A.plot_acc(C_values,poly_accuracies,linear_accuracies,rbf_accuracies)


# Use the best kernel and C value to predict the test dataset
kernel = 'rbf'
C = 5
test_pred, test_acc= A.SVM_model(Pn_train_flat, Pn_train_labels, Pn_test_flat, Pn_test_labels, kernel, C)

#visualise the prediction result and use confusion matrix to analysis the result
A.visualize_pred( Pn_test, Pn_test_labels, test_pred, 9, 'visualize')

A.con_matrix(Pn_test_labels, test_pred, 'test_pred')

#flattent and normolize the image data to the range from 0 to 1 (28x28x1) for CNN model
Pn_train_images = A.CNN_flatten(Pn_train)
Pn_val_images = A.CNN_flatten(Pn_val)
Pn_test_images = A.CNN_flatten(Pn_test)

#set up and compile a CNN model
Pn_CNN_model = A.CNN_model()
Pn_CNN_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# test validation set if needed
# Pn_CNN_val = Pn_CNN_model.fit(Pn_train_images, Pn_train_labels, steps_per_epoch=147 ,epochs=30, validation_data=(Pn_val_images, Pn_val_labels), validation_steps=15)
# A.CNN_result(Pn_CNN_val)

#test dataset into CNN model
Pn_CNN_test = Pn_CNN_model.fit(Pn_train_images, Pn_train_labels, steps_per_epoch=147 ,epochs=30, validation_data=(Pn_test_images, Pn_test_labels), validation_steps=19, batch_size = 32 )
#show the loss and accuracy
A.CNN_result(Pn_CNN_test)

#Flatten the PathMNIST data for Random Forest model
Pa_train_flat = B.RF_flatten(Pa_train)
Pa_val_flat = B.RF_flatten(Pa_val)
Pa_test_flat = B.RF_flatten(Pa_test)

#set up Random Forest Model and get prediction
RF_test_pred, RF_test_acc = B.RF_model(Pa_train_flat, Pa_train_labels, Pa_test_flat, Pa_test_labels)

#use confusion matrix to analysis the result
B.con_matrix(Pa_test_labels, RF_test_pred, 'RF_confusion_matrix')

#flatten the data for CNN model of PathMNIST
Pa_train_images = B.CNN_flatten(Pa_train)
Pa_val_images = B.CNN_flatten(Pa_val)
Pa_test_images = B.CNN_flatten(Pa_test)

#set up and compile a CNN model
Pa_CNN_model = B.CNN_model()
Pa_CNN_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# test validation set if needed
# val_history = Pa_CNN_model.fit(Pa_train_images, Pa_train_labels, steps_per_epoch=1000 ,epochs=30, validation_data=(Pa_val_images, Pa_val_labels), validation_steps=100, batch_size=32)
# B.CNN_result(val_history,'validation')

#train and test part of the dataset to save time
test_history = Pa_CNN_model.fit(Pa_train_images, Pa_train_labels, steps_per_epoch=1000 ,epochs=30, validation_data=(Pa_test_images, Pa_test_labels), validation_steps=100, batch_size=32)
#show the loss and accuracy
B.CNN_result(test_history,'test')