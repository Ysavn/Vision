import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_differential_filter():
    # To do
    filter_x = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return filter_x, filter_y


def filter_image(im, filter):
    # To do
    rows = im.shape[0] - filter.shape[0] + 1
    cols = im.shape[1] - filter.shape[1] + 1
    im_filtered = np.zeros((rows, cols))
    for i in range(0, rows):
        for j in range(0, cols):
            im_filtered[i][j] = np.sum(im[i:i+3, j:j+3]*filter)
    return im_filtered


def get_gradient(im_dx, im_dy):
    # To do
    rows = im_dx.shape[0]
    cols = im_dx.shape[1]
    grad_mag = np.zeros((rows, cols))
    grad_angle = np.zeros((rows, cols))
    for i in range(0, rows):
        for j in range(0, cols):
            grad_mag[i][j] = (im_dx[i][j]**2 + im_dy[i][j]**2)**(0.5)
            grad_angle[i][j] = (np.arctan2(im_dy[i][j], im_dx[i][j]) + np.pi)%(np.pi)

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    rows = grad_mag.shape[0]
    cols = grad_mag.shape[1]
    depth = 6
    ori_histo = np.zeros((grad_mag.shape[0]//cell_size, grad_mag.shape[1]//cell_size, depth))
    for i in range(rows):
        for j in range(cols):
            if i//cell_size >= ori_histo.shape[0] or j//cell_size >= ori_histo.shape[1]:
                continue
            angle = grad_angle[i][j] * (180/np.pi)
            idx = int(((angle//15 + 1)//2)%6)
            ori_histo[i//cell_size][j//cell_size][idx]+=grad_mag[i][j]
    return ori_histo

def get_block_descriptor(ori_histo, block_size):
    # To do
    rows = ori_histo.shape[0] - (block_size - 1)
    cols = ori_histo.shape[1] - (block_size - 1)
    ori_histo_normalized  = np.zeros((rows, cols, 24))
    for i in range(rows):
        for j in range(cols):
            v = np.asarray([])
            for k in range(block_size):
                for l in range(block_size):
                    v = np.hstack((v, ori_histo[i+k][j+l]))
            v = v / (np.sum(v**2) + 0.001**2)**0.5
            ori_histo_normalized[i][j] = v
    return ori_histo_normalized

def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    im = np.pad(im, ((1, 1), (1, 1)), 'constant')
    # To do
    filter_x, filter_y = get_differential_filter()
    filtered_image_x = filter_image(im, filter_x)
    filtered_image_y = filter_image(im, filter_y)
    grad_mag, grad_angle = get_gradient(filtered_image_x, filtered_image_y)
    ori_histo = build_histogram(grad_mag, grad_angle, 8)
    hog = get_block_descriptor(ori_histo, 2)

    #f, ax = plt.subplots(1, 2)
    #ax[0].title.set_text("Gradient Magnitude")
    #ax[1].title.set_text("Gradient Angle")
    #ax[0].imshow(filtered_image_x, cmap=plt.cm.gray)
    #ax[1].imshow(filtered_image_y, cmap=plt.cm.gray)
    #plt.show()
    # visualize to verify
    #visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def normalize(hog):
    mean = np.mean(hog)
    hog = hog - mean
    return hog

def calculate_iou(bounding_box1, bounding_box2, box_size):
    x1 = bounding_box1[0]
    y1 = bounding_box1[1]

    x2 = bounding_box2[0]
    y2 = bounding_box2[1]

    length = max(0, box_size - np.abs(x1-x2))
    breadth = max(0, box_size - np.abs(y1-y2))
    area = length*breadth
    return area / (box_size**2 + box_size**2 - area)


def non_maximum_suppression(bounding_boxes, box_size):

    new_bounding_boxes = np.zeros((0, 3))
    while bounding_boxes.shape[0] > 0:
        k = np.argmax(bounding_boxes[:, [2]])
        new_bounding_boxes = np.append(new_bounding_boxes, [bounding_boxes[k]], axis=0)
        tmp = np.zeros((0, 3))
        for j in range(bounding_boxes.shape[0]):
            iou = calculate_iou(bounding_boxes[k], bounding_boxes[j], box_size)
            #print(iou)
            if iou < 0.5:
                tmp = np.append(tmp, [bounding_boxes[j]], axis=0)
        bounding_boxes = tmp
    return new_bounding_boxes

def face_recognition(I_target, I_template):

    stride = 1
    template_hog = extract_hog(I_template)
    template_hog = normalize(template_hog)
    template_norm = np.linalg.norm(template_hog)

    rows = I_target.shape[0] - I_template.shape[0] + 1
    cols = I_target.shape[1] - I_template.shape[1] + 1
    boxes = np.zeros((rows, cols))

    for i in range(0, rows, stride):
        for j in range(0, cols, stride):
            patch = I_target[i:i+I_template.shape[0], j:j+I_template.shape[1]]
            patch_hog = normalize(extract_hog(patch))
            boxes[i][j] = np.sum(np.multiply(patch_hog, template_hog))/(np.linalg.norm(patch_hog)*template_norm)

    bounding_boxes = np.empty((0, 3))
    for i in range(boxes.shape[0]):
        for j in range(boxes.shape[1]):
            if boxes[i][j] > 0.46:
               bounding_boxes = np.append(bounding_boxes, np.asarray([[j, i, boxes[i][j]]]), axis = 0)

    bounding_boxes = non_maximum_suppression(bounding_boxes, I_template.shape[0])

    return  bounding_boxes


def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        print(x1, y1, x2, y2, bounding_boxes[ii][2])
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()



if __name__=='__main__':

    im = cv2.imread('img1.tif', 0)
    #plt.imshow(im, cmap=plt.cm.gray)
    #plt.axis('off')
    #plt.show()
    hog = extract_hog(im)

    I_target= cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template

    bounding_boxes = face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    #this is visualization code.



