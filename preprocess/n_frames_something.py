from __future__ import print_function, division
import os
import sys
import subprocess

def class_process(dir_path, class_name):
  class_path = os.path.join(dir_path, class_name)
  if not os.path.isdir(class_path):
    return

  image_indices = []
  for image_file_name in os.listdir(class_path):
    if 'image' not in image_file_name:
      continue
    image_indices.append(int(image_file_name[6:11]))

  if len(image_indices) == 0:
    print('no image files', class_path)
    n_frames = 0
  else:
    image_indices.sort(reverse=True)
    n_frames = image_indices[0]
    print(class_path, n_frames)
  with open(os.path.join(class_path, 'n_frames'), 'w') as dst_file:
    dst_file.write(str(n_frames))


if __name__=="__main__":
  dir_path = sys.argv[1]
  for class_name in os.listdir(dir_path):
    class_process(dir_path, class_name)
