import cropper
import numpy as np
import os

#runs all of the 13 boards and splits them for our testing data (2925 tiles)
def crop_all_images():
    #turns the data directory into a list of all board pngs
    all_boards = os.listdir("data")

    print(f"length:{len(all_boards)}")
    print(all_boards[1])
    print(f"./data/{all_boards[0]}")
    #loops through all boards in the list and crops all of them
    for board in range(len(all_boards)):
        path = f"./data/{all_boards[board]}"
        print(f"path: {path}")
        cropper.run_cropper(path)
        

crop_all_images()