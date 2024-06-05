import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def create_simple_image():

    width, height = 100, 100
    image = Image.new('L', (width, height), color=0)  # 'L' mode is for grayscale


    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 30, 30], fill=255)
    draw.rectangle([40, 40, 60, 60], fill=255)
    draw.rectangle([70, 10, 90, 30], fill=255)
    draw.rectangle([10, 70, 30, 90], fill=255)


    image_path = 'simple_image.png'
    image.save(image_path)
    print(f"Image saved as {image_path}")

    return image_path


def component_labeling_4_connectivity(image):
    rows, cols = image.shape
    labels = np.zeros_like(image, dtype=int)
    label = 1
    equivalence = {}

    for i in range(rows):
        for j in range(cols):
            if image[i][j] == 1:
                left_label = labels[i][j - 1] if j > 0 else 0
                upper_label = labels[i - 1][j] if i > 0 else 0

                if left_label == 0 and upper_label == 0:
                    labels[i][j] = label
                    equivalence[label] = label
                    label += 1
                elif left_label > 0 and upper_label == 0:
                    labels[i][j] = left_label
                elif upper_label > 0 and left_label == 0:
                    labels[i][j] = upper_label
                elif left_label == upper_label:
                    labels[i][j] = left_label
                else:
                    labels[i][j] = upper_label
                    equivalence[left_label] = upper_label

    for k in equivalence.keys():
        root = equivalence[k]
        while equivalence[root] != root:
            root = equivalence[root]
        equivalence[k] = root

    for i in range(rows):
        for j in range(cols):
            if labels[i][j] != 0:
                labels[i][j] = equivalence[labels[i][j]]

    return labels


def component_labeling_8_connectivity(image, min_intensity=0, max_intensity=255):
    rows, cols = image.shape
    labels = np.zeros_like(image, dtype=int)
    label = 1
    equivalence = {}

    for i in range(rows):
        for j in range(cols):
            if min_intensity <= image[i][j] <= max_intensity:
                neighbors = []

                if j > 0:
                    neighbors.append(labels[i][j - 1])
                if i > 0:
                    neighbors.append(labels[i - 1][j])
                if i > 0 and j > 0:
                    neighbors.append(labels[i - 1][j - 1])
                if i > 0 and j < cols - 1:
                    neighbors.append(labels[i - 1][j + 1])

                neighbors = [n for n in neighbors if n > 0]
                if not neighbors:
                    labels[i][j] = label
                    equivalence[label] = label
                    label += 1
                else:
                    min_label = min(neighbors)
                    labels[i][j] = min_label
                    for n in neighbors:
                        if n != min_label:
                            equivalence[n] = min_label


    print(f"Equivalence before resolving: {equivalence}")

    for k in equivalence.keys():
        root = equivalence[k]
        while equivalence[root] != root:
            root = equivalence[root]
        equivalence[k] = root


    print(f"Equivalence after resolving: {equivalence}")

    for i in range(rows):
        for j in range(cols):
            if labels[i][j] != 0:
                labels[i][j] = equivalence[labels[i][j]]

    return labels


def size_filter(labeled_image, size_threshold):
    sizes = np.bincount(labeled_image.ravel())
    mask = sizes > size_threshold
    mask[0] = False

    filtered_image = labeled_image.copy()
    for label, keep in enumerate(mask):
        if not keep:
            filtered_image[filtered_image == label] = 0

    return filtered_image


def load_and_process_image(image_path, binary_threshold=128):
    image = Image.open(image_path)
    grayscale_image = image.convert('L')
    binary_image = np.array(grayscale_image) > binary_threshold
    binary_image = binary_image.astype(int)
    return binary_image


def display_images(images, titles):
    num_images = len(images)
    plt.figure(figsize=(16, 8))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image, cmap='nipy_spectral' if np.max(image) > 1 else 'gray')
        if np.max(image) > 1:
            plt.colorbar()
        plt.title(title)
        plt.axis('off')
    plt.show()


def main():

    image_path = create_simple_image()

    binary_image = load_and_process_image(image_path)
    grayscale_image = np.array(Image.open(image_path).convert('L'))

    labeled_image_4_conn = component_labeling_4_connectivity(binary_image)

    min_intensity = int(input("Enter minimum intensity value for 8-connectivity: "))
    max_intensity = int(input("Enter maximum intensity value for 8-connectivity: "))
    labeled_image_8_conn = component_labeling_8_connectivity(grayscale_image, min_intensity, max_intensity)

    size_threshold = int(input("Enter size threshold for filtering components: "))
    filtered_image = size_filter(labeled_image_8_conn, size_threshold)

    display_images(
        [binary_image, labeled_image_4_conn, grayscale_image, labeled_image_8_conn, filtered_image],
        [
            "Binary Image",
            "Labeled Image (4-connectivity)",
            "Original Grayscale Image",
            "Labeled Image (8-connectivity)",
            "Filtered Image"
        ]
    )

if __name__ == "__main__":
    main()
