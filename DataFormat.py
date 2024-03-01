import shutil
import random
import os

base_path = r'C:\Users\PC\OneDrive\Desktop\sign language detection\DATA'
os.chdir(base_path)
print("Current Working Directory:", os.getcwd())

if not os.path.isdir('train/A'):
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')

alph = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

for i in alph:
    os.mkdir(os.path.join('train', i))
    os.mkdir(os.path.join('valid', i))
    os.mkdir(os.path.join('test', i))

    source_path = os.path.join(base_path, i)
    print("Source Path:", source_path)

    try:
        train_files = os.listdir(source_path)
    except FileNotFoundError:
        print(f"Directory not found: {source_path}")
        continue

    valid_samples = random.sample(train_files, min(50, len(train_files)))
    test_samples = random.sample([file for file in train_files if file not in valid_samples], min(10, len(train_files) - 50))

    for j in valid_samples:
        shutil.move(os.path.join(source_path, j), os.path.join('valid', i, j))

    for j in test_samples:
        shutil.move(os.path.join(source_path, j), os.path.join('test', i, j))

    for j in train_files:
        if j not in valid_samples and j not in test_samples:
            shutil.move(os.path.join(source_path, j), os.path.join('train', i, j))

os.chdir('..')