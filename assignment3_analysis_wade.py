from assignment3_initial_code_wade import generate_dataset, generate_model
import tensorflow as tf
import numpy as np
from pathlib import Path
import shutil
import keras

# Change these to reflect wherever you put your data:
SYNTHETIC_DATA_DIR = "./image_classification/data/synthetic"
TEST_DATA_DIR = "./image_classification/data/real_graphs"
# Set our constants to what we used in our model generation code
IMG_HEIGHT = 150
IMG_WIDTH = 150

# How many test images we're running predictions on:
NUM_TEST_IMAGES=40

# Generate the dataset:
generate_dataset(data_dir=SYNTHETIC_DATA_DIR)

# Generate the model:
model = generate_model(SYNTHETIC_DATA_DIR)

# Now let's check our quality:

# We'll load in our reference images as a dataset
check_ds = keras.utils.image_dataset_from_directory(
    TEST_DATA_DIR,
    shuffle=False,
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size=NUM_TEST_IMAGES
)

# Grab the class names (should be drawn from the directory)
real_class_names = check_ds.class_names
# Grab the file names too, might be useful for checking why some images do better than others
real_file_names = [Path(file).name for file in check_ds.file_paths]

# Now, run the predictions using our model!
predictions = model.predict(check_ds, verbose=0)
# Calculate the scores
scores = tf.nn.softmax(predictions, axis=1)
# And what class each image is predicted to belong to
predicted_class_indexes= np.argmax(scores, axis=1)

# We'll keep track of how many how many we get right:
predicted_correctly = 0

# Because our batch size include all the images, we only need to take one batch -
# if we had multiple batches, we'd need more
for images, labels in check_ds.take(1):
    # for every image in the dataset
    for i in range(len(images)):
        # Grab the name of the file and the class it is catagorized as:
        image_name = real_file_names[i]
        real_class = real_class_names[labels[i]]

        # Figure out what class our model has predicted it belongs to:
        predicted_class = real_class_names[predicted_class_indexes[i]]

        # If we predicted correctly, up our correct predictions!
        if real_class == predicted_class:            
            predicted_correctly += 1

        print(f"{image_name} is {real_class} and was classifed as {predicted_class}")

# Print out our final results:
print(f"During this run, we scored {predicted_correctly} of {NUM_TEST_IMAGES} images correctly.")
print(f"We got {round(predicted_correctly/NUM_TEST_IMAGES,2) * 100}% of them correct.")

# Note that there is some randomness here, so we probably want to do this several times
# and average it all out to get a real idea how a given model is performing.

