#by Avneet Singh Saluja (5586107)
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors


def find_match(img1, img2):
    # To do
    sift = cv2.xfeatures2d.SIFT_create()
    neigh = NearestNeighbors()
    neigh2 = NearestNeighbors()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    neigh.fit(des2)
    x1 = np.empty(shape=[0, 2])
    x2 = np.empty(shape=[0, 2])
    keypoint_set = set()
    for i in range(des1.shape[0]):
        dist, indx = neigh.kneighbors([des1[i]], n_neighbors=2)
        if dist[0][0]/dist[0][1] < 0.7:
            keypoint_set.add((i, indx[0][0]))

    neigh2.fit(des1)
    for i in range(des2.shape[0]):
        dist, indx = neigh2.kneighbors([des2[i]], n_neighbors=2)
        if dist[0][0] / dist[0][1] < 0.7 and (indx[0][0], i) in keypoint_set:
            x2 = np.append(x2, [[kp2[i].pt[0], kp2[i].pt[1]]], axis=0)
            x1 = np.append(x1, [[kp1[indx[0][0]].pt[0], kp1[indx[0][0]].pt[1]]], axis=0)
    return x1, x2

def show_inliers_keypoints(max_x, x1, x2, ransac_thr):
    A = np.zeros(shape=[x1.shape[0] * 2, 6])
    b = np.empty(shape=[x1.shape[0] * 2, 1])
    x1_ = np.empty(shape=[0, 2])
    x2_ = np.empty(shape=[0, 2])
    for j in range(x1.shape[0]):
        A[j * 2][2] = A[j * 2 + 1][5] = 1
        A[j * 2][0] = A[j * 2 + 1][3] = x1[j][0]
        A[j * 2][1] = A[j * 2 + 1][4] = x1[j][1]
        b[j * 2][0] = x2[j][0]
        b[j * 2 + 1][0] = x2[j][1]
    b_ = np.dot(A, max_x)
    for i in range(x1.shape[0]):
        err = ((b_[i*2][0] - b[i*2][0]) ** 2 + (b_[i*2 + 1][0] - b[i*2 + 1][0]) ** 2) ** 0.5
        if err < ransac_thr:
            x1_ = np.append(x1_, [[x1[i][0], x1[i][1]]], axis = 0)
            x2_ = np.append(x2_, [[x2[i][0], x2[i][1]]], axis = 0)
    return x1_, x2_

def calculate_inliers(x1, x2, x_, ransac_thr):

    A = np.zeros(shape=[x1.shape[0]*2, 6])
    b = np.empty(shape=[x1.shape[0]*2, 1])
    cnt_inliers = 0
    for j in range(x1.shape[0]):
        A[j * 2][2] = A[j * 2 + 1][5] = 1
        A[j * 2][0] = A[j * 2 + 1][3] = x1[j][0]
        A[j * 2][1] = A[j * 2 + 1][4] = x1[j][1]
        b[j * 2][0] = x2[j][0]
        b[j * 2 + 1][0] = x2[j][1]
    b_ = np.dot(A, x_)
    for i in range(x1.shape[0]):
        err = ((b_[i*2][0] - b[i*2][0])**2 + (b_[i*2+1][0]-b[i*2+1][0])**2)**0.5
        #print(err)
        if err < ransac_thr:
            #print(err, ransac_thr)
            cnt_inliers += 1
    return cnt_inliers

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    max_inliers = 0
    A_ = np.zeros(shape=[6, 6])
    for i in range(ransac_iter):
        b = np.empty(shape=[6, 1])
        smpls_indx = random.sample(range(x1.shape[0]), 3)
        for j, indx in enumerate(smpls_indx):
            b[j*2][0] = x2[indx][0]
            b[j*2+1][0] = x2[indx][1]
            A_[j*2][2] = A_[j*2+1][5] = 1
            A_[j*2][0] = A_[j*2+1][3] = x1[indx][0]
            A_[j*2][1] = A_[j*2+1][4] = x1[indx][1]
        x_ = np.dot(np.dot(np.linalg.pinv(np.dot(A_.transpose(), A_)), A_.transpose()), b)
        cnt_inliers = calculate_inliers(x1, x2, x_, ransac_thr)
        if cnt_inliers > max_inliers:
            max_x = x_
            max_inliers = cnt_inliers

    A = np.eye(3)
    for i in range(max_x.shape[0]):
        A[i//3][i%3] = max_x[i][0]
    return A

def withinImageSize(x, output_size):
    if x[0][0] >=0 and x[0][0] < output_size[1] and x[1][0] >=0 and x[1][0] < output_size[0]:
        return True
    return False

def warp_image(img, A, output_size):
    # To do
    img_warped = np.full(shape=[output_size[0], output_size[1]], fill_value=-1.0)
    x = np.ones(shape=[3, 1])
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            x[0][0] = j
            x[1][0] = i
            x_ = np.matmul(A, x)
            if withinImageSize(x_, img.shape):
                img_warped[i][j]=img[int(x_[1][0])][int(x_[0][0])]
    return img_warped

def getSobelFilter():
    filter_x = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return filter_x, filter_y

def filter_image(im, filter):
    rows = im.shape[0] - filter.shape[0] + 1
    cols = im.shape[1] - filter.shape[1] + 1
    im_filtered = np.zeros((rows, cols))
    for i in range(0, rows):
        for j in range(0, cols):
            im_filtered[i][j] = np.sum(im[i:i+3, j:j+3]*filter)
    return im_filtered

def calculateImageGradient(img):
    img = np.pad(img, ((1, 1), (1, 1)), 'constant')
    filter_x, filter_y = getSobelFilter()
    filtered_img_x = filter_image(img, filter_x)
    filtered_img_y = filter_image(img, filter_y)
    gradient_img = np.empty(shape=[filtered_img_x.shape[0], filtered_img_x.shape[1], 1, 2])
    for i in range(gradient_img.shape[0]):
        for j in range(gradient_img.shape[1]):
            gradient_img[i][j][0][0] = filtered_img_x[i][j]
            gradient_img[i][j][0][1] = filtered_img_y[i][j]
    return gradient_img

def calculateJacobian(img):
    jacob =  np.zeros(shape=[img.shape[0], img.shape[1], 2, 6])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            jacob[i][j][0][0] = jacob[i][j][1][3] = j
            jacob[i][j][0][1] = jacob[i][j][1][4] = i
            jacob[i][j][0][2] = jacob[i][j][1][5] = 1
    return jacob

def calculateSteepestDescent(grad, jacob):
    steepest_descent = np.empty(shape=[jacob.shape[0], jacob.shape[1], 1, 6])
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            steepest_descent[i][j] = np.matmul(grad[i][j], jacob[i][j])
    return steepest_descent

def calculateHessian(steepest_descent):
    hessian = np.zeros(shape=[6, 6])
    for i in range(steepest_descent.shape[0]):
        for j in range(steepest_descent.shape[1]):
            hessian = np.add(hessian, np.matmul(np.transpose(steepest_descent[i][j]), steepest_descent[i][j]))
    return hessian

def get_magnitude(p):
    return np.sum(p ** 2) ** 0.5

def calculateF(steepest_descent, error):
    F = np.zeros(shape=[6, 1])
    for i in range(error.shape[0]):
        for j in range(error.shape[1]):
            F = np.add(F, np.transpose(steepest_descent[i][j])*error[i][j])
    return F


def align_image(template, target, A):
    # To do
    p = np.empty(shape=[6, 1])
    for i in range(p.shape[0]):
        p[i] = A[i//3][i%3]
    p[0] -= 1
    p[4] -= 1
    template_grad = calculateImageGradient(template)
    template_jacob = calculateJacobian(template)
    template_steepest_descent = calculateSteepestDescent(template_grad, template_jacob)
    errors = np.asarray([])
    hessian = calculateHessian(template_steepest_descent)
    itr = 0
    delta_p = np.ones(shape=[6, 1])
    while np.linalg.norm(delta_p) > (10**(-3)) and itr < 1000:
        target_warped = warp_image(target, A, template.shape)
        error = target_warped - template
        errors = np.append(errors, (np.sum(error**2)**0.5))
        F = calculateF(template_steepest_descent, error)
        delta_p = np.matmul(np.linalg.pinv(hessian), F)
        A_delta_p = np.eye(3)
        for i in range(6):
            A_delta_p[i//3][i%3]+=delta_p[i]
        A = np.matmul(A, np.linalg.pinv(A_delta_p))
        itr+=1
        #print(itr, np.linalg.norm(delta_p))
    A_refined = A
    return A_refined, errors/np.linalg.norm(errors)


def track_multi_frames(template, img_list):
    # To do
    A_list = []
    x1, x2 = find_match(template, img_list[0])
    A = align_image_using_feature(x1, x2, 5, 5000)
    for i in range(len(img_list)):
        A_refined, errors = align_image(template, img_list[i], A)
        A_list.append(A_refined)
        #visualize_align_image(template, img_list[i], A, A_refined, errors)
        template = warp_image(img_list[i], A_refined, template.shape).astype(np.uint8)
        A = A_refined
    return A_list

def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors*255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    A = align_image_using_feature(x1, x2, 5, 5000)
    #x1_, x2_ = show_inliers_keypoints(max_x, x1, x2, 5)
    #visualize_find_match(template, target_list[0], x1_, x2_)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    #A_refined, errors = align_image(template, target_list[0], A)
    #visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)