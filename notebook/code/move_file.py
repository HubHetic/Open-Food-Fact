
import os
import shutil
import sys


def move(path_file, new_path):
    if os.path.exists(path_file) and os.path.exists(new_path):
        lis_img = os.listdir(path_file)
        for im in lis_img:
            shutil.move(path_file + im, new_path)


if __name__ == "__main__":
    move(sys.argv[1], sys.argv[2])
