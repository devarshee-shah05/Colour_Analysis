import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76


# Function definitions

def RGB2HEX(color):
  return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(image_data):
  image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image

def get_colors(image, number_of_colors):
  modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
  modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

  clf = KMeans(n_clusters=number_of_colors, random_state=42)
  labels = clf.fit_predict(modified_image)

  counts = Counter(labels)
  counts = dict(sorted(counts.items()))

  center_colors = clf.cluster_centers_
  ordered_colors = [center_colors[i] for i in counts.keys()]
  hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]

  return list(zip(hex_colors, counts.values()))

# Streamlit UI

st.title('Colour Analysis for better ranking')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  # Read image from file uploader
  image = get_image(uploaded_file.read())

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