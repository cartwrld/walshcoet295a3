# Imports and function definitions
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import random
from pathlib import Path
import keras
from keras import layers
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam

from charts import acc_chart, loss_chart

# Set some constants we'll use in all functions
IMG_HEIGHT = 150
IMG_WIDTH = 150

# I don't want to generate a completely different set of graphs each time,
# so to reduce that randomness I'm going to set our random seed.
random.seed(124)

# We don't need to display all these graphs:
plt.ioff()

colors = ['green', 'blue', 'red', 'purple', 'brown', 'black']


def set_plot(x_data, y_data, type):
    fig = plt.figure()
    # fig = plt.figure(figsize=(random.randint(3,10),random.randint(3,7)))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_data, y_data, color=colors[random.randint(0, len(colors) - 1)], linewidth=random.randint(1, 5))
    ax.set_title(f"{type} Graph")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.canvas.draw()
    my_array = np.asarray(fig.canvas.buffer_rgba())
    plt.close()
    return my_array


def generate_empty_graph():
    """
        Generates an empty graph image
        returns: An PIL image version of an empty graph
        rtype: Image
    """
    fig = plt.figure()
    #fig = plt.figure(figsize=(random.randint(3,10),random.randint(3,7)))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Empty Graph")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.canvas.draw()
    my_array = np.asarray(fig.canvas.buffer_rgba())
    plt.close()
    my_image = Image.fromarray(my_array)
    return my_image


def generate_linear_graph():
    """
        Generates a linear graph image
        returns: An PIL image version of a linear graph
        rtype: Image
    """
    x_series = np.arange(random.randint(-10, 10), random.randint(10, 50))
    a = random.random() * 3
    b = random.random() * 100
    y_series = x_series * a + b

    my_image = Image.fromarray(set_plot(x_series, y_series, "Linear"))
    return my_image


def generate_quadratic_graph():
    """
    Generates a quadratic graph image
    returns: An PIL image of a quadratic graph
    rtype: Image
    """
    x_series = np.arange(random.randint(-10, 10), random.randint(10, 50))
    a = random.random() * 3
    b = random.random() * 100
    y_series = (a * x_series ** 2) + b

    my_image = Image.fromarray(set_plot(x_series, y_series, "Quadratic"))
    return my_image


def generate_trigonometric_graph():
    """
    Generates a trigonometric graph image
    returns: An PIL image of an trigonometric graph
    rtype: Image
    """
    x_series = np.arange(random.randint(-10, 10), random.randint(10, 50))
    a = random.random() * 3
    b = random.random() * 100
    # y_series = (a * np.sin(x_series)) + b
    y_series = np.cos(x_series)

    my_image = Image.fromarray(set_plot(x_series, y_series, "Trigonometric"))
    return my_image


def generate_dataset(data_dir="data/synthetic"):
    """Creates a dataset of synthetic graph images in the data_dir supplied"""

    # Create the directories if they don't already exist
    Path(f"{data_dir}/linear/").mkdir(exist_ok=True, parents=True)
    Path(f"{data_dir}/quadratic/").mkdir(exist_ok=True, parents=True)
    Path(f"{data_dir}/trigonometric/").mkdir(exist_ok=True, parents=True)
    Path(f"{data_dir}/empty/").mkdir(exist_ok=True, parents=True)

    # Create 100 images of each type
    for i in range(1, 200):
        generate_empty_graph().save(Path(f"{data_dir}/empty/empty{i}.png"), "PNG")
        generate_linear_graph().save(Path(f"{data_dir}/linear/linear{i}.png"), "PNG")
        generate_quadratic_graph().save(Path(f"{data_dir}/quadratic/quadratic{i}.png"), "PNG")
        if i % 2 == 0:
            # pass
            generate_trigonometric_graph().save(Path(f"{data_dir}/trigonometric/trigonometric{i}.png"), "PNG")


def generate_model(bs, vs, eps, data_dir="data/synthetic", ):
    """
    Generates a tensorflow model from the data in the supplied data directory
    returns: An trained model for identifying graph types
    rtype: Sequential
    """

    # Standard batch size and validation split:
    BATCH_SIZE = bs
    VALIDATION_SPLIT = vs
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=124,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, verbose=0
    )

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=124,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, verbose=0
    )
    # Grab the class names before we do the performance tweaks otherwise we get issues
    class_names = train_ds.class_names

    # Standard performance tweaks
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Assemble our model
    model = Sequential([
        layers.Input((IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Rescaling(1. / 255),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        # layers.Dropout(0.5),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        # layers.Dropout(0.4),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        # layers.Dropout(0.3),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        # layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        # layers.Dense(512, activation='relu'),
        layers.Dense(len(class_names))
    ])

    # Compile the model
    model.compile(optimizer=Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # print(model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model for 10 epochs
    epochs = eps
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping], verbose=0

    )

    acc_chart(history)
    loss_chart(history)

    return model
