import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from medmnist import PneumoniaMNIST
import random
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


#PRE-PROCESSING DATA

#download the file
def download_Pn_data():
    
    PneumoniaMNIST(split="train", download=True)
    PneumoniaMNIST(split="val", download=True)
    PneumoniaMNIST(split="test", download=True)

    return

# #load and split the data
def load_data(path):

    Pn = np.load(path)

    Pn_train = Pn['train_images']
    Pn_val = Pn['val_images']
    Pn_test = Pn['test_images']
    Pn_train_labels = Pn['train_labels']
    Pn_val_labels = Pn['val_labels']
    Pn_test_labels = Pn['test_labels']
    
    return Pn, Pn_train, Pn_val, Pn_test, Pn_train_labels, Pn_val_labels, Pn_test_labels

#count the number of normal and pneumonia
def count_pneumonia(data):

    normal_count = np.sum(data == 0)  # '0' is the label for normal
    pneumonia_count = np.sum(data == 1)  # '1' is the label for pneumonia

    return normal_count, pneumonia_count

#distribution of normal and pneumonia bar plot
def distr_plot(train_normal, train_pneumonia, val_normal, val_pneumonia,test_normal, test_pneumonia, name):

    plot_path = 'AMLS_23-24_SN23187901/A'
    full_path = f"{plot_path}/{name}"
    plt.figure(figsize=(10, 6))
    categories = ['train_normal', 'train_pneumonia', 'val_normal', 'val_pneumonia', 'test_normal', 'test_pneumonia']
    # Counts
    counts = [train_normal, train_pneumonia, val_normal, val_pneumonia, test_normal ,test_pneumonia]
    # Create bar plot
    plt.bar(categories, counts)
    # plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Distribution of Normal and Pneumonia Cases')
    plt.savefig(full_path)

    return

#flatten the data for SVM model
def SVM_flatten(data):
    
    flatten_data = data.reshape(data.shape[0], -1)/ 255.0

    return flatten_data

#flatten the data for CNN model
def CNN_flatten(data):
    
    flatten_data = data.reshape(data.shape[0], 28, 28, 1)/ 255.0

    return flatten_data

#MODEL DEFINATION

#define SVM model
def SVM_model(X_train, y_train, X_test, y_test, kernel, C):

    svm_clf = SVC(C = C, kernel = kernel)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return y_pred, accuracy

#define CNN model
def CNN_model():

    model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='sigmoid')
    ])

    return model

#Plot SVM accuracy
def plot_acc(C_values,poly_accuracies,linear_accuracies,rbf_accuracies):
    
    plot_path = 'AMLS_23-24_SN23187901/A'
    full_path = f"{plot_path}/SVM_val_acc"

    plt.figure()
    plt.plot(C_values, poly_accuracies,label='poly')
    plt.plot(C_values, linear_accuracies,label='linear')
    plt.plot(C_values, rbf_accuracies,label='rbf')
    plt.xscale('log')
    plt.xlabel('C value')
    plt.ylabel('Accuracy')
    plt.title('SVM Accuracy for Pneumonia Prediction Based on C Value')
    plt.legend()
    plt.savefig(full_path)

    return

#RESULT VISUALIZATION AND ANALISYS

def con_matrix(real_label, pred_label, name):

    plot_path = 'AMLS_23-24_SN23187901/A'
    full_path = f"{plot_path}/{name}"
    cm = confusion_matrix(real_label, pred_label)

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(full_path)

    return


def visualize_pred( X_test, y_test, selected_pred, num_images, name):

    selected_indices = random.sample(range(len(X_test)), num_images)
    selected_images = X_test[selected_indices]
    selected_labels = y_test[selected_indices]

    plot_path = 'AMLS_23-24_SN23187901/A'
    full_path = f"{plot_path}/{name}"

    fig, axes = plt.subplots(3, num_images // 3, figsize=(15, 12))

    for i, ax in enumerate(axes.flatten()):

        ax.imshow(selected_images[i], cmap='gray')
        ax.set_title('Prediction: Pneumonia' if selected_pred[i] == 1 else 'Prediction: normal',fontsize=20)
        ax2 = ax.twinx()
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title('Pneumonia' if selected_labels[i] == 1 else 'Normal', fontsize=20, pad=20)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(full_path)

    return      

#show CNN model result
def CNN_result(data):

    plot_path = 'AMLS_23-24_SN23187901/A'
    acc_path = f"{plot_path}/{'accuracy vs epochs'}"
    loss_path = f"{plot_path}/{'loss vs epochs'}"

    acc = data.history['accuracy']
    val_acc = data.history['val_accuracy']
    loss = data.history['loss']
    val_loss = data.history['val_loss']

    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, label='training accuracy')
    plt.plot(epochs, val_acc, label='Val accuracy')
    plt.legend()
    plt.savefig(acc_path)

    plt.figure()
    plt.plot(epochs, loss, label='training loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.legend()
    plt.savefig(loss_path)

    return
