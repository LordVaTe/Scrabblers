import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os
from datetime import datetime
import pathlib
import shutil

def filter_close_corners(corners, min_distance=30):
    if not corners:
        return []

    corners_array = np.array(corners)
    filtered_corners = []

    for corner in corners_array:
        if all(np.linalg.norm(corner - other_corner) >= min_distance for other_corner in filtered_corners):
            filtered_corners.append(corner)

    return filtered_corners


def extend_line(p1,p2,img_shape):
    rows,cols = img_shape[:2]
    x1,y1 = p1
    x2,y2 = p2

    if x2 == x1:
        return (x1,0), (x1, rows-1)

    m = (y2-y1)/(x2-x1)
    c = y1 - m * x1

    if m == 0:
        return (0,y1), (cols-1,y1)

    x0 = int(-c/m) if m != 0 else 0
    y0 = 0
    xf = cols - 1
    yf = int(m*xf + c)

    p1 = (max(min(x0,cols-1),0), max(min(y0,rows-1),0))
    p2 = (max(min(xf,cols-1),0), max(min(yf,rows-1),0))

    return p1, p2

def line_intersection(line1,line2):
    rho1,theta1 = line1
    rho2,theta2 = line2
    A = np.array([
        [np.cos(theta1),np.sin(theta1)],
        [np.cos(theta2),np.sin(theta2)]
    ])
    b = np.array([[rho1],[rho2]])

    det = np.linalg.det(A)
    if np.isclose(det, 0):
        return None

    x0, y0 = np.linalg.solve(A,b)
    return int(np.round(x0)),int(np.round(y0))

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def split_into_grid(img, pdt, pdb, pdl, pdr):
    w = (img.shape[1] - pdl - pdr) / 15
    h = (img.shape[0] - pdt - pdb) / 15
    
    cells = []
    for row in range(15):
        for col in range(15):
            start_x = int(pdl + col * w)
            end_x = int(start_x + w)
            start_y = int(pdt + row * h)
            end_y = int(start_y + h)
            
            cell = img[start_y:end_y, start_x:end_x]
            cells.append(cell)
    return cells

dir_list = []
def save_to_dir(images, base_dir="split_cells/run_"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f"{base_dir}{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    
    x_labels = [chr(i) for i in range(ord('A'), ord('O') + 1)]
    y_labels = [str(i) for i in range(1, 16)]

    all_files = []
    for i, img in enumerate(images):
        x_coord = x_labels[i % 15]
        y_coord = y_labels[i // 15]
        filename = os.path.join(dir_name, f"cell_{x_coord}{y_coord}.png")
        all_files.append(filename)
        cv2.imwrite(filename, img)
        print(f"Saved {filename}")
    
    #checks two cells for if they are exactly the same which they are if cropping fails
    first_img = cv2.imread(all_files[0])
    second_img = cv2.imread(all_files[1])
    difference = cv2.subtract(first_img,second_img)
    b,g,r = cv2.split(difference)

    #deletes the bad cropping 
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print(f"Error cropping {all_files[0]}")
        print(f"Deleting bad directory: {dir_name}")
        shutil.rmtree(dir_name)
    else:
        dir_list.append(dir_name)
        print(f"Cropping Successful")

def test_data_folder():
    destination = "split_cells/test_data"
    shutil.move(dir_list[0],destination)
    
    

    


# img = cv2.imread('./data/test_cropped.png')
# img = cv2.imread('./data/test_empty.png')
# img = cv2.imread('./data/test_emptyslant.png')
# img = cv2.imread('./data/test_full.png')

# img = cv2.imread('./data/19-1.png')
# img = cv2.imread('./data/19-2.png')
# img = cv2.imread('./data/19-3.png')
# img = cv2.imread('./data/19-4.png')
# img = cv2.imread('./data/19-5.png')

# img = cv2.imread('./data/22-1.png')
# img = cv2.imread('./data/fs1.png')


def crop_img(fpath):
    img = cv2.imread(fpath)
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(5,5),0)
    _, thresholded = cv2.threshold(blur,0,255,cv2.THRESH_BINARY +cv2.THRESH_OTSU)
    cntrs, _ = cv2.findContours(thresholded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    lgst_cntr = max(cntrs,key=cv2.contourArea)
    mask = np.zeros_like(grey)
    cv2.drawContours(mask,[lgst_cntr],-1,255,-1)
    edges = cv2.Canny(mask,50,150)
    lines = cv2.HoughLines(edges,1,np.pi/180,threshold=100)

    line_img = np.zeros_like(img)
    intersections = []
    if lines is not None:
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho

                p1 = (int(x0+1000*(-b)), int(y0+1000*(a)))
                p2 = (int(x0-1000*(-b)), int(y0-1000*(a)))
                p1,p2 = extend_line(p1,p2,img.shape)

                cv2.line(line_img,p1,p2,(0,255,0),2)
                intersections.append((rho,theta))

    corners = []
    for i, line1 in enumerate(intersections):
        for line2 in intersections[i+1:]:
            inter = line_intersection(line1,line2)
            if inter is not None:
                corners.append(inter)

    epsilon = 100
    min_samples = 1
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(corners)
    labels = db.labels_

    clusters = []

    for label in np.unique(labels):
        if label == -1:
            continue

        labelMask = (labels==label)
        cluster = np.array(corners)[labelMask]

        centroid = cluster.mean(axis=0).astype("int")
        clusters.append((centroid, cluster))
        cv2.circle(img, tuple(centroid), 6, (0, 0, 255), -1)

    clusters.sort(key=lambda x: len(x[1]), reverse=True)

    if len(clusters) >= 4:
        corner_clusters = clusters[:4]
        final_corners = [c[0] for c in corner_clusters]
        for x, y in final_corners:
            cv2.circle(img, (x, y), 8, (255, 0, 0), -1)
    else:
        print("not enough clus to find corners")

    cv2.imshow('corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    corners = np.array(final_corners)

    ordered_corners = order_points(corners)

    widthA = np.sqrt(((ordered_corners[2][0]-ordered_corners[3][0])**2) + ((ordered_corners[2][1]-ordered_corners[3][1])**2))
    widthB = np.sqrt(((ordered_corners[1][0]-ordered_corners[0][0])**2) + ((ordered_corners[1][1]-ordered_corners[0][1])**2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((ordered_corners[1][0]-ordered_corners[2][0])**2) + ((ordered_corners[1][1]-ordered_corners[2][1])**2))
    heightB = np.sqrt(((ordered_corners[0][0]-ordered_corners[3][0])**2) + ((ordered_corners[0][1]-ordered_corners[3][1])**2))
    maxHeight = max(int(heightA), int(heightB))

    dest = np.array([
        [0, 0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]],
        dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_corners,dest)
    warped = cv2.warpPerspective(img,M,(maxWidth,maxHeight))

    cv2.imshow('cropped',warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return warped

# for others: this is padding for the edges of the board - 
# I don't think it will be necessary to change but just fyi
pdt = 28
pdb = 28
pdl = 28
pdr = 28


# previews of a select few split images
# def demo():
#     cv2.imshow('Cell A1 (tl corner)', cells[0])
#     cv2.imshow('Cell G8 (center)', cells[7 * 15 + 7])
#     cv2.imshow('Cell O15 (br corner)', cells[14 * 15 + 14])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
# this is the image here   vvvvvvv   which is going to be split
def run_cropper(filepath):
    cropped = crop_img(filepath)
    cells = split_into_grid(cropped, pdt, pdb, pdl, pdr)
    save_to_dir(cells)
    
#run_cropper("./data/19-1.png")
