import csv
import clip
import torch
from PIL import Image
import numpy as np
import tqdm


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)


x = []
y = []
n_images = 0
with open("dataset_moda_7.csv", 'r') as file:
    csvreader = csv.reader(file, delimiter=',')
    headers = next(csvreader)
    for row in csvreader:
        n_images += 1
        image_url, nota = row

        image_name = image_url.split("/")[-1]
        image_path = "dataset/" + image_name

        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)

        im_emb_arr = image_features.cpu().detach().numpy()
        im_emb_arr = normalized(im_emb_arr)

        x.append(im_emb_arr)

        y_ = np.zeros((1, 1))

        y_[0][0] = nota

        y.append(y_)

        if n_images % 1000 == 0:
            print(n_images)

x = np.vstack(x)
y = np.vstack(y)
print(x.shape)
print(y.shape)
np.save('x_ModaDataset_CLIP_L14_embeddings.npy', x)
np.save('y_ratings_7.npy', y)

