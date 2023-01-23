import os
import cv2
import re
import numpy as np
from scipy import stats as st
from datetime import datetime


def extract_color(img, mask):
    '''
    img and mask must have the same shape, mask has 1 channel
    mask: 0 - 255
    '''
    if mask.shape != img.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    img = np.reshape(img, (-1, 3))
    mask = mask.flatten()
    color_arr = img[mask > 120]

    if len(color_arr) == 0:
        return np.int0([0, 0, 0])

    r = np.median(color_arr[:, 0])
    g = np.median(color_arr[:, 1])
    b = np.median(color_arr[:, 2])

    return np.int0([r, g, b])


class TextDistributor:
    def __init__(self, roi_lengths, paragraph):
        self.roi_lengths = roi_lengths
        self.paragraph = re.sub(' +', ' ', paragraph.strip())
        # remove space between punctuation and words
        self.paragraph = re.sub(r'\s([?.!",](?:\s|$))', r'\1', self.paragraph)
        # remove space between quotation marks and words
        self.paragraph = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', self.paragraph)
        self.segmented_text, penalty = self.run(0, 0, 0, [])
        print("Penalty", penalty)

    def run(self, start_idx, penalty, roi_idx, result):
        if roi_idx >= len(self.roi_lengths):
            return result, penalty
        limit = self.roi_lengths[roi_idx]

        if start_idx >= len(self.paragraph):
            return result, penalty + limit
        text = self.paragraph[start_idx:].strip()
        offset = len(self.paragraph) - start_idx - len(text)

        if roi_idx == len(self.roi_lengths) - 1 or len(text) <= limit or len(text.split()) == 1:
            result.append(text)
            penalty += abs(len(text) - limit)
            return result, penalty

        # cut before the limit
        # find index of the nearest space
        space_idx = None
        for i in range(limit, 1, -1):
            if text[i] == " ":
                space_idx = i
                break

        if space_idx is None:
            left_result = result
            left_penalty = 1e9
        else:
            left_text = text[:space_idx].strip()
            left_result, left_penalty = self.run(start_idx + space_idx + offset,
                                                 penalty + abs(limit - len(left_text)),
                                                 roi_idx + 1, result + [left_text])

        # cut after the limit
        space_idx = None
        for i in range(limit, len(text)):
            if text[i] == " ":
                space_idx = i
                break
        if space_idx is None:
            right_result = result
            right_penalty = 1e9
        else:
            right_text = text[:space_idx].strip()
            right_result, right_penalty = self.run(start_idx + space_idx + offset,
                                                   penalty + abs(limit - len(right_text)),
                                                   roi_idx + 1, result + [right_text])

        if left_penalty < right_penalty:
            return left_result, left_penalty

        return right_result, right_penalty


if __name__ == "__main__":
    img = cv2.imread("./inputs/street9/original.png")
    mask = cv2.imread("./inputs/street9/mask.png", cv2.IMREAD_GRAYSCALE)
    color = extract_color(img, mask)
    print(color)
