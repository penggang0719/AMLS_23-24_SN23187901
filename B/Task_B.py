import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from medmnist import PathMNIST
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#PRE-PROCESSING DATA

#download the file
def download_Pa_data():
    
    PathMNIST(split="train", download=True)
    PathMNIST(split="val", download=True)
    PathMNIST(split="test", download=True)

    return

# load and split the data
def load_data(path):

    Pa = np.load(path)

    Pa_train = Pa['train_images']
    Pa_val = Pa['val_images']
    Pa_test = Pa['test_images']
    Pa_train_labels = Pa['train_labels']
    Pa_val_labels = Pa['val_labels']
    Pa_test_labels = Pa['test_labels']
    
    return Pa, Pa_train, Pa_val, Pa_test, Pa_train_labels, Pa_val_labels, Pa_test_labels


#flatten the data for Random Forest model
def RF_flatten(data):
    
    flatten_data = data.reshape(data.shape[0], -1)/ 255.0

    return flatten_data

#flatten the data for CNN model
def CNN_flatten(data):
    
    flatten_data = data.reshape(data.shape[0], 28, 28, 3)/ 255.0

    return flatten_data

#MODEL DEFINATION
#define Random Forest model
def RF_model(X_train, y_train, X_test, y_test):

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    return predictions, accuracy

#define CNN model
def CNN_model():

    model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(9, activation='softmax')
    ])

    return model

def con_matrix(real_label, pred_label, name):

    plot_path = 'AMLS_23-24_SN23187901/B'
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

#show CNN model result
def CNN_result(data,name):

    # plot_path = 'AMLS_23-24_SN23187901/B'
    plot_path = 'AMLS_23-24_SN23187901/B'
    acc_path = f"{plot_path}/{name} accuracy vs epochs"
    loss_path = f"{plot_path}/{name} loss vs epochs"

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
