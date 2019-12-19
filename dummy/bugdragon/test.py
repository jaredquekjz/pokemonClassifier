from PIL import Image
import os

directory = r'./'
c = 1
print(os.getcwd())
for filename in os.listdir():
    if filename.endswith(".png"):
        im = Image.open(filename)
        name = filename+'.jpg'
        rgb_im = im.convert('RGB')
        rgb_im.save(name)
        c += 1
        print(os.path.join(directory, filename))
        continue
    else:
        continue
