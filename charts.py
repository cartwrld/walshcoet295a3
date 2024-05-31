import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                   #
#   Name:            Carter Walsh (walsh0715)       #
#   Class:           COET295 - Assignment 2         #
#   Instructor:      Bryce Barrie & Wade Lahoda     #
#   Date:            Monday, May 27th, 2024         #
#                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # #

def acc_chart(results, title=""):
    # plt.title("Accuracy of Model")
    plt.figure(figsize=(10,10))
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.legend(['train', 'test'], loc="upper left")
    plt.title(title + " Accuracy")
    plt.show()


def loss_chart(results, title=""):
    # plt.title("Model Losses")
    plt.figure(figsize=(10,10))
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.legend(['train', 'test'], loc="upper left")
    plt.title(title + " Losses")
    plt.show()
