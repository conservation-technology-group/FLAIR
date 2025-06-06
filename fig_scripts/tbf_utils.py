# Utils for calculating tailbeat frequency from a series of shark masks

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gc
import cv2
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([1, 0, 0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# find the center of mass, head, and tail positions and display them
def annotate_shark_parts(mask, ax):
    # convert the mask to binary format and ensure single channel
    if mask.ndim > 2:
        mask_binary = mask[0, :, :]  # take one channel if it's multi-channel
    else:
        mask_binary = mask
 
    mask_binary = (mask_binary > 0).astype(np.uint8) * 255 # ensure the mask is in binary format (0 and 255)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find the contours of the mask

    if contours:
        main_contour = max(contours, key=cv2.contourArea) # get the largest contour (assuming it's the shark's body)
        moments = cv2.moments(main_contour) # calculate the center of mass (COM) of the mask
        if moments["m00"] != 0:
            com_x = int(moments["m10"] / moments["m00"])
            com_y = int(moments["m01"] / moments["m00"])
            com = np.array([com_x, com_y])

            # find the two furthest points on the contour
            max_distance = 0
            furthest_points = (None, None)
            for i in range(len(main_contour)):
                for j in range(i + 1, len(main_contour)):
                    pt1 = tuple(main_contour[i][0])
                    pt2 = tuple(main_contour[j][0])
                    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
                    if dist > max_distance:
                        max_distance = dist
                        furthest_points = (pt1, pt2)

            # identify head and tail based on distance to COM
            if furthest_points[0] and furthest_points[1]:
                dist1 = np.linalg.norm(np.array(furthest_points[0]) - com)
                dist2 = np.linalg.norm(np.array(furthest_points[1]) - com)
                # the point further from COM is the tail (blue); the other is the head (black)
                tail = np.array(furthest_points[0]) if dist1 > dist2 else np.array(furthest_points[1])
                head = np.array(furthest_points[1]) if dist1 > dist2 else np.array(furthest_points[0])

                # compute direction vector for COM to head
                head_direction = head - com
                head_norm = head_direction / np.linalg.norm(head_direction)

                # compute vector from COM to tail
                tail_vector = tail - com

                # project tail_vector onto head_direction (COM to Head direction vector)
                projection_length = np.dot(tail_vector, head_norm)
                projection_point = com + projection_length * head_norm

                # calculate perpendicular distance
                perp_distance = np.linalg.norm(tail - projection_point)

                # determine the sign of the perpendicular distance using the cross product
                cross_product_z = np.cross(
                    np.append(head_direction, 0),  # Append 0 for z-dimension
                    np.append(tail_vector, 0)
                )[2]  # Extract z-component

                signed_distance = perp_distance * np.sign(cross_product_z)

                # draw the perpendicular line from tail to the projection point
                # ax.plot([tail[0], projection_point[0]], [tail[1], projection_point[1]], 'c-', label="Tail-Perpendicular (Cyan)")

                # print the signed perpendicular distance to verify
                return signed_distance

def smooth_data(data, window_length=15, polyorder=2):
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)

# find zero crossings to identify tailbeats
def find_zero_crossings(data):
    zero_crossings = np.where(np.diff(np.sign(data)))[0]
    return zero_crossings


def refine_zero_crossings(data, zero_crossings, extrema):
    refined_crossings = []
    all_extrema = np.sort(extrema)

    # ensure first zero-crossing is captured
    if zero_crossings[0] < all_extrema[0]:
        refined_crossings.append(zero_crossings[0])

    # refine zero-crossings between extrema
    for i in range(len(all_extrema) - 1):
        start, end = all_extrema[i], all_extrema[i + 1]
        crossing_in_range = zero_crossings[(zero_crossings > start) & (zero_crossings < end)]
        if len(crossing_in_range) > 0:
            refined_crossings.append(crossing_in_range[0])  # take the first valid crossing

    # ensure last zero-crossing is captured
    if zero_crossings[-1] > all_extrema[-1]:
        refined_crossings.append(zero_crossings[-1])

    return np.array(refined_crossings)

def calculate_tailbeat_periods(data, frame_rate=30, prominence_factor=0.05):
    smoothed_data = smooth_data(data) # smooth the data
    zero_crossings = find_zero_crossings(smoothed_data) # find zero crossings

    # find peaks and troughs
    peaks, _ = find_peaks(smoothed_data, prominence=prominence_factor * np.ptp(smoothed_data))
    troughs, _ = find_peaks(-smoothed_data, prominence=prominence_factor * np.ptp(smoothed_data))

    # combine peaks and troughs
    all_extrema = np.sort(np.concatenate((peaks, troughs)))

    # refine zero crossings
    refined_crossings = refine_zero_crossings(smoothed_data, zero_crossings, all_extrema)

    # calculate periods using refined crossings
    tailbeat_periods = np.diff(refined_crossings)
    tailbeat_periods_seconds = tailbeat_periods / frame_rate

    return tailbeat_periods, tailbeat_periods_seconds, smoothed_data, refined_crossings, all_extrema
