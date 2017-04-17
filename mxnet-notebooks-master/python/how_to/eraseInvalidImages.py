import os

root = './data/images'
valid_paths = []
with open('data/valid_ids.txt', 'rb') as f:
    for uid in f:
        path = os.path.join(root, uid.strip()) + '.jpg'
        valid_paths.append(path)        

print valid_paths

for filename in os.listdir(root):
    path = os.path.join(root, filename)
    if path not in valid_paths:
        os.remove(path)  # remove the file
