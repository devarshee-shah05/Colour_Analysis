from PIL import Image
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76


# Function definitions

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_image(image_data):
    image = Image.open(image_data)
    image = image.convert('RGB')  # Ensure RGB mode
    return np.array(image)  # Convert to NumPy array for processing


def get_colors(image, number_of_colors):
    # Resize the image (optional)
    # resized_image = image.resize((600, 400), Image.ANTIALIAS)
    # image = np.array(resized_image)

    # Flatten the image (similar to OpenCV reshape)
    modified_image = image.flatten().reshape(image.shape[0] * image.shape[1], 3)

    # KMeans clustering
    clf = KMeans(n_clusters=number_of_colors, random_state=42)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(color) for color in ordered_colors]

    return list(zip(hex_colors, counts.values()))


# Streamlit UI

st.title('Colour Analysis for better ranking')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image from file uploader
    image = get_image(uploaded_file)

    # Display image using st.image()
    st.image(image, caption='Uploaded Image', use_column_width=True)

    number_of_colors = st.slider('Select number of colors', min_value=1, max_value=20, value=5)


if st.button('Identify Colors'):
    result = get_colors(image, number_of_colors)

    st.write("### Color Distribution")

    # Create and display the table with custom display text
    data = [["*Percentage", "Hex Color*"]]  # First row for headers
    data.extend([["{:.1f}%".format(count / sum([count for _, count in result]) * 100), hex_color] for hex_color, count in result])
    table = st.table(data)

    # Display pie chart
    fig = plt.figure(figsize=(12, 8))
    explode = [0.1] * len(result)  # Example: Explode all slices equally
    plt.pie([count for _, count in result], labels=[hex for hex, _ in result], colors=[hex for hex, _ in result], explode=explode, autopct='%1.1f%%')
    st.pyplot(fig)
