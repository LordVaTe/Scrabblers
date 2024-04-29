import cropper
import numpy as np
import os

#runs all of the 13 boards and splits them for our testing data (2925 tiles)
def crop_all_images(fpath):
    #turns the data directory into a list of all board pngs
    all_boards = os.listdir("data")
    #loops through all boards in the list and crops all of them
    for board in range(len(all_boards)):
        #current croped board
        cropper.cropped(board)
