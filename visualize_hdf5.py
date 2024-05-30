import h5py
from moviepy.editor import ImageSequenceClip
from PIL import Image
import io
import numpy as np
import os

hdf5_paths = []

directories = os.listdir("data")
for directory in directories:
    if not os.path.isdir(f"data/{directory}"):
        continue
    file = os.listdir(f"data/{directory}")[0]
    hdf5_paths.append(f"data/{directory}/{file}")
    
print(len(hdf5_paths))

for path in hdf5_paths:
    with h5py.File(path, 'r') as f:
        save_path = f'{path[:-5]}.mp4'
        instruction_path = f'{path[:-5]}.txt'
        frames = []
        instruction = f['instruction'][()].decode('utf-8')
        for image in f['observations']['images']['cam_high']:
            img = Image.open(io.BytesIO(image))
            rgb_img = img.convert('RGB')
            frames.append(rgb_img)
        clip = ImageSequenceClip([np.array(frame) for frame in frames], fps=24)
        clip.write_videofile(save_path, codec='libx264')
        print(f'Saved video to {save_path}')
        with open(instruction_path, 'w') as f2:
            f2.write(instruction)
        print(f'Saved instruction to {instruction_path}')
