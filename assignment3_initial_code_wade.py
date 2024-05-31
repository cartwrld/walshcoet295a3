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

# Set some constants we'll use in all functions
IMG_HEIGHT = 150
IMG_WIDTH = 150

# I don't want to generate a completely different set of graphs each time,
# so to reduce that randomness I'm going to set our random seed.
random.seed(123)

# We don't need to display all these graphs:
plt.ioff()

colors = ['green', 'blue', 'red', 'purple', 'brown', 'black']


def generate_empty_graph():
    """
    Generates an empty graph image
    returns: An PIL image version of an empty graph
    rtype: Image
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Linear Graph")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
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
    x_series = x_series = np.arange(1, 20)
    a = random.random() * 3
    b = random.random() * 100
    y_series = x_series * a + b

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_series, y_series, color=colors[random.randint(0, len(colors) - 1)], linewidth=random.randint(1, 5))
    ax.set_title("Linear Graph")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    fig.canvas.draw()
    my_array = np.asarray(fig.canvas.buffer_rgba())
    plt.close()
    my_image = Image.fromarray(my_array)
    return my_image


def generate_quadratic_graph():
    """
    Generates a quadratic graph image
    returns: An PIL image of a quadratic graph
    rtype: Image
    """
    x_series = x_series = np.arange(1, 20)
    a = random.random() * 3
    b = random.random() * 100
    y_series = (a * x_series ** 2) + b

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_series, y_series, color=colors[random.randint(0, len(colors) - 1)], linewidth=random.randint(1, 5))
    ax.set_title("Quadratic Graph")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    fig.canvas.draw()
    my_array = np.asarray(fig.canvas.buffer_rgba())
    plt.close()
    my_image = Image.fromarray(my_array)
    return my_image


def generate_trigonometric_graph():
    """
    Generates a trigonometric graph image
    returns: An PIL image of an trigonometric graph
    rtype: Image
    """
    x_series = np.linspace(0, 4 * np.pi, 100)  # Generates 100 points from 0 to 4π
    amplitude = random.uniform(0.5, 3.0)  # Amplitude between 0.5 and 3
    frequency = random.uniform(0.5, 2.0)  # Frequency between 0.5 and 2
    phase_shift = random.uniform(0, 2 * np.pi)  # Phase shift between 0 and 2π
    vertical_shift = random.uniform(-10, 10)  # Vertical shift between -10 and 10

    # Generate the y_series using a randomized sine function
    y_series = amplitude * np.sin(frequency * x_series + phase_shift) + vertical_shift

    fig, ax = plt.subplots()
    ax.plot(x_series, y_series)
    ax.set_title("Trigonometric Graph")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    fig.canvas.draw()
    my_array = np.array(fig.canvas.buffer_rgba())
    plt.close(fig)

    my_image = Image.fromarray(my_array)
    return my_image


def generate_dataset(data_dir="data/synthetic"):
    """Creates a dataset of synthetic graph images in the data_dir supplied"""

    # Create the directories if they don't already exist
    Path(f"{data_dir}/linear/").mkdir(exist_ok=True, parents=True)
    Path(f"{data_dir}/quadratic/").mkdir(exist_ok=True, parents=True)
    Path(f"{data_dir}/trigonometric/").mkdir(exist_ok=True, parents=True)
    Path(f"{data_dir}/empty/").mkdir(exist_ok=True, parents=True)

    # Create 100 images of each type
    for i in range(1, 100):
        generate_empty_graph().save(Path(f"{data_dir}/empty/empty{i}.png"), "PNG")
        generate_linear_graph().save(Path(f"{data_dir}/linear/linear{i}.png"), "PNG")
        generate_quadratic_graph().save(Path(f"{data_dir}/quadratic/quadratic{i}.png"), "PNG")
        if i % 6 == 0:
            generate_trigonometric_graph().save(Path(f"{data_dir}/trigonometric/trigonometric{i}.png"), "PNG")


def generate_model(bs, vs, eps, data_dir="data/synthetic"):
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
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        verbose=0
    )

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        verbose=0
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
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(len(class_names))
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # print(model.summary())

    # Train the model for 10 epochs
    epochs = eps
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs, verbose=0
    )

    return model








