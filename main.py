from PIL import Image, ImageDraw
import math
import skimage
import streamlit as st
import skimage.io
from skimage.morphology import binary_closing
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.exposure
import numpy as np
from sklearn import tree
import skimage.io
import skimage.filters
import skimage.morphology
import joblib

WINDOW_SIZE = 5
MODEL_FILE = "model.pkl"


def binarize(pixel_value):
    if pixel_value < 128:
        return 0
    else:
        return 255

def compare(image1, image2):

    arr1 = np.array(image1.convert('L').point(binarize))
    arr2 = np.array(image2.convert('L').point(binarize))
    dec = ["TP", "TN", "FP", "FN"]
    lista = []

    new_image = Image.new('RGB', (arr1.shape[1], arr1.shape[0]), (0, 0, 0))
    draw = ImageDraw.Draw(new_image)

    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[1]):
            if arr1[i][j] == 0 and arr2[i][j] == 0:
                lista.append(dec[0])
                draw.point((j, i), fill=(0, 0, 0))
            elif arr1[i][j] == 255 and arr2[i][j] == 255:
                lista.append(dec[1])
                draw.point((j, i), fill=(255, 255, 255))
            elif arr1[i][j] == 0 and arr2[i][j] == 255:
                lista.append(dec[2])
                draw.point((j, i), fill=(0, 255, 0))
            elif arr1[i][j] == 255 and arr2[i][j] == 0:
                lista.append(dec[3])
                draw.point((j, i), fill=(255, 0, 0))
    accuracy = (lista.count("TP") + lista.count("TN"))/len(lista)
    sensitivity = lista.count("TP") / (lista.count("TP")+lista.count("FN"))
    specificity = (lista.count("TN") / (lista.count("FP") + lista.count("TN")))
    precision = lista.count("TP") / (lista.count("TP") + lista.count("FP"))
    G_Mean = math.sqrt(sensitivity*specificity)
    F_measure = (2*precision*sensitivity)/(precision+sensitivity)
    data = [accuracy, sensitivity, specificity, precision, G_Mean, F_measure]
    return new_image, data

def load_data(image_files, label_files):
    images = []
    labels = []
    for image_file, label_file in zip(image_files, label_files):
        image = np.array(Image.open(image_file))
        label = np.array(Image.open(label_file).convert("1"))
        images.append(image)
        labels.append(label)
    return images, labels

def preprocess_data(images, labels):

    features = []
    labels_flat = []

    for image, label in zip(images, labels):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if i >= WINDOW_SIZE // 2 and i < image.shape[0] - WINDOW_SIZE // 2 and \
                        j >= WINDOW_SIZE // 2 and j < image.shape[1] - WINDOW_SIZE // 2:
                    window = image[i - WINDOW_SIZE // 2:i + WINDOW_SIZE // 2 + 1,
                             j - WINDOW_SIZE // 2:j + WINDOW_SIZE // 2 + 1, 1]
                    feature = window.flatten()
                    features.append(feature)
                    labels_flat.append(label[i, j])

    features = np.array(features)
    labels_flat = np.array(labels_flat)
    return features, labels_flat

def train_decision_tree(image_files, label_files):
    images, labels = load_data(image_files, label_files)
    features, labels_flat = preprocess_data(images, labels)
    clf = tree.DecisionTreeClassifier()
    clf.fit(features, labels_flat)
    joblib.dump(clf, MODEL_FILE)
    return clf


def detect_blood_vessels(image_file, clf):
    image = np.array(Image.open(image_file))
    padded_image = np.pad(image, ((WINDOW_SIZE // 2, WINDOW_SIZE // 2), (WINDOW_SIZE // 2, WINDOW_SIZE // 2), (0, 0)),
                          mode='constant')
    features = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + WINDOW_SIZE, j:j + WINDOW_SIZE, 1]
            feature = window.flatten()
            features.append(feature)
    features = np.array(features)
    predicted_labels = clf.predict(features)
    predicted_labels = predicted_labels.reshape(image.shape[:2])
    predicted_labels = skimage.morphology.remove_small_objects(predicted_labels, min_size=100, connectivity=1)
    predicted_labels = skimage.morphology.remove_small_holes(predicted_labels, area_threshold=1000,
                                                             connectivity=10000, out=predicted_labels)
    predicted_labels = skimage.morphology.binary_dilation(predicted_labels, out=None)
    predicted_labels = skimage.morphology.binary_erosion(predicted_labels, out=None)
    output_image = Image.fromarray((predicted_labels * 255).astype(np.uint8))
    return output_image

image_files = ["01_h.jpg", "02_h.jpg", "03_h.jpg"]
label_files = ["01_h.tif", "02_h.tif", "03_h.tif"]

def main():
    try:
        clf = joblib.load(MODEL_FILE)
        print("Loaded saved model")
    except FileNotFoundError:
        clf = train_decision_tree(image_files, label_files)
        print("Trained and saved model")

    new_image_file = "08_h.jpg"
    output_image = detect_blood_vessels(new_image_file, clf)
    im3, lista = compare(Image.open("08_h.tif"), output_image)

    # Tytuł strony
    st.title("Badanie dna siatkowki oka")
    st.header("Obrazy:")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(Image.open("08_h.jpg"), use_column_width=True, caption="Original")
    with col2:
        st.image(Image.open("08_h.tif"), use_column_width=True, caption="Blood vessels")
    with col3:
        st.image(output_image, use_column_width=True, caption="Machine learning")
    with col4:
        st.image(im3, use_column_width=True, caption="Comparison")
    r1, r2 = st.columns(2)
    with r1:
        st.write("Accuracy: ", lista[0])
        st.write("Sensitivity: ", lista[1])
        st.write("Specificity: ", lista[2])
    with r2:
        st.write("Precision: ", lista[3])
        st.write("G_Mean: ", lista[4])
        st.write("F_measure: ", lista[5])

main()