import os
import pandas as pd

image_folder = '/root/autodl-tmp/test'

image_names = []

for root, dirs, files in os.walk(image_folder):
    for file in files:
        
        if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_names.append(file)

df = pd.DataFrame(image_names, columns=['chart'])
df.to_csv('/root/autodl-tmp/test.csv', index=False)

print(f"All image names have been saved to the test.csv file, containing {len(image_names)} images.")
