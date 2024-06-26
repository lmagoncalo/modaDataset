import csv
import requests

n_images = 0
with open("dataset_moda_7.csv", 'r') as file:
    csvreader = csv.reader(file, delimiter=',')
    headers = next(csvreader)
    for row in csvreader:
        image_url, nota = row

        image_name = image_url.split("/")[-1]
        image_name = "dataset/" + image_name
        # print(image_name)

        img_data = requests.get(image_url).content
        with open(image_name, 'wb') as handler:
            handler.write(img_data)

        n_images += 1

        if n_images % 200 == 0:
            print(n_images)

print("Total:", n_images)
