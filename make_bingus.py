import cv2
import numpy as np
from keras.models import load_model
from typing import Tuple, List


def get_latent_vectors(n_samples, latent_dim):
    latent_vectors = np.random.randn(latent_dim*n_samples)
    latent_vectors = latent_vectors.reshape(n_samples, latent_dim)
    return latent_vectors


def generate_images(n: int, generator) -> np.ndarray:
    vec = get_latent_vectors(n, 100)
    images = generator.predict(vec)
    images = (images + 1) * (255 / 2)
    images = images.astype(np.uint8)
    return images


def classify_breed(image: np.ndarray, classifier, limit: float = 0.1,
                   max_breeds: int = 2) -> List[Tuple[str, np.float32]]:
    categories = (
        "Bambino", "Don Sphynx",
        "Donskoy", "Dwelf",
        "Elf", "Minskin",
        "Peterbald", "Sphynx",
        "Ukrainian Levkoy",
    )
    image = image.astype(np.float16)
    image = np.expand_dims(image, axis=0)
    image = (1/255) * image
    prediction = classifier.predict(image)[0]
    sorted_indexes = np.argsort(prediction)
    sorted_indexes = np.flip(sorted_indexes)
    breed_percentage = []
    percent_sum = 1
    for index in sorted_indexes[:max_breeds]:
        if prediction[index] < limit:
            break
        breed_percentage.append((categories[index], prediction[index]))
        percent_sum -= prediction[index]
    breed_percentage.append(("Other", percent_sum))
    return breed_percentage


generator = load_model("models/BingusX/G_BingusX60")
classifier = load_model("models/C_BingusX")

for _ in range(4):
    image = generate_images(1, generator)[0]
    breeds = classify_breed(image, classifier)
    title = " ~ ".join((f"{name} [{prob*100:.0f}%]" for name, prob in breeds))
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    cv2.imshow(title, image)
    cv2.waitKey(0)
