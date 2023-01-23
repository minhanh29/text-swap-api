import os
from time import time
import textwrap
from textblob import TextBlob
import math
from skimage import io
from scipy import stats
from scipy.signal import wiener
import cv2
import argparse
import torch
from tqdm import tqdm
import numpy as np
from utils import TextDistributor, extract_color
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from lama import Inpainter
from googletrans import Translator
import onnxruntime

import io
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./keys/text-replace-366410-b1df0306b203.json"
from google.cloud import vision

img_dir = "./inputs/assasin_title/"
height = 64
pil_to_tensor = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Grayscale(1)
])

# for testing
write_image = True


def preprocess(batch):
    img_batch = []

    w_sum = 0
    for item in batch:
        h, w = item.shape[1:]
        scale_ratio = height / h
        w_sum += int(w * scale_ratio)

    to_h = height
    to_w = w_sum
    to_w = int(round(to_w / 8)) * 8
    to_w = max(to_h, to_w)
    to_scale = (to_h, to_w)
    torch_resize = transforms.Resize(to_scale)
    cnt = 0
    for img in batch:
        img = torch_resize(img)
        img_batch.append(img)
        cnt += 1

    img_batch = torch.stack(img_batch)

    return img_batch


def expand_bbox(bbox, img_shape, w_ratio=0.1, h_ratio=0.3):
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1
    padding_w = max(int(w * w_ratio), 10)
    padding_h = max(int(h * h_ratio), 10)

    x1 -= padding_w
    x2 += padding_w
    y1 -= padding_h
    y2 += padding_h

    x1 = int(max(0, min(x1, img_shape[1]-1)))
    x2 = int(max(0, min(x2, img_shape[1]-1)))
    y1 = int(max(0, min(y1, img_shape[0]-1)))
    y2 = int(max(0, min(y2, img_shape[0]-1)))

    return x1, y1, x2, y2


class ModelFactory:
    def __init__(self, model_dir="./weights", font_dir="./fonts"):
        self.font_dir = font_dir
        FONT_FILE = "./font_list.txt"
        with open(FONT_FILE, "r", encoding="utf-8") as f:
            font_list = f.readlines()
        self.font_list = [f.strip().split("|")[-1] for f in font_list]

        self.device = torch.device("cpu")
        self.mask_net = onnxruntime.InferenceSession(os.path.join(model_dir, "mask_net.onnx"))
        self.font_clf = onnxruntime.InferenceSession(os.path.join(model_dir, "font_classifier.onnx"))

        self.inpainter = Inpainter(os.path.join(model_dir, "lama-fourier"))

        self.K = torch.nn.ZeroPad2d((0, 1, 1, 0))
        self.my_pad = torch.nn.ZeroPad2d((3, 3, 3, 3))

        self.exclude_list = np.array([147, 38, 39, 14, 15, 127, 128, 129, 91, 223])

        self.translator = Translator()
        self.ocr_client = vision.ImageAnnotatorClient()

    def translate(self, text):
        return self.translator.translate(text, src="en", dest="vi").text

    def detect_text(self, img_path):
        # Loads the image into memory
        with io.open(img_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = self.ocr_client.text_detection(image=image)
        texts = response.text_annotations
        print('Texts:')

        result = []
        for i, text in enumerate(texts):
            print(text.description)
            if i == 0:
                continue

            vertices = ([[vertex.x, vertex.y]
                        for vertex in text.bounding_poly.vertices])

            print("bounds", vertices)
            boxes = np.array(vertices)
            result.append([boxes, text.description])
        return result

    def extract_background(self, img_list, mask_list):
        new_img_list = []
        new_mask_list = []
        for img, mask in zip(img_list, mask_list):
            # img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

            # Lama
            print("Image shape", img.shape)
            img = np.transpose(img, (2, 0, 1))
            img = img.astype('float32') / 255
            mask = mask.astype('float32') / 255

            new_img_list.append(img)
            new_mask_list.append(mask)

        result = self.inpainter.predict(new_img_list, new_mask_list)
        return result
        # return new_img_list

    def extract_mask(self, img):
        img = img/127.5 - 1
        img = preprocess(img)
        mask_s = self.mask_net.run(None, {self.mask_net.get_inputs()[0].name: img.cpu().numpy()})[0]
        mask_s = torch.from_numpy(mask_s)
        mask_s = self.K(mask_s)
        return mask_s

    def detect_font(self, img):
        img = preprocess(img)
        pred = self.font_clf.run(None, {self.font_clf.get_inputs()[0].name: img.cpu().numpy()})[0]
        pred[:, self.exclude_list] = 0
        chosen = np.argmax(pred, axis=-1)
        font_path = []
        for idx in chosen:
            font_path.append(os.path.join(self.font_dir, self.font_list[int(idx)]))
        return font_path

class BBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def get(self):
        return self.x1, self.y1, self.x2, self.y2

    def crop(self, img):
        return img[self.y1:self.y2, self.x1:self.x2]

    def abs_inner(self, innerBox):
        x1, y1, x2, y2 = innerBox.get()
        return BBox(self.x1 + x1, self.y1 + y1, self.x1 + x2, self.y1 + y2)

    def width(self):
        return abs(self.x2 - self.x1)

    def height(self):
        return abs(self.y2 - self.y1)

    def center_point(self):
        return np.array([(self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2])

    def __str__(self):
        return f"({self.x1}, {self.y1}, {self.x2}, {self.y2})"

    def __repr__(self):
        return self.__str__()


class WordRoi:
    def __init__(self, bbox, text):
        self.text = text
        self.bbox = bbox

class Roi:
    def __init__(self, id, img, innerBbox, bbox, text, model_factory, center_align=True):
        self.id = id
        self.text = text
        self.target_text = ""
        self.center_align = center_align
        self.color = np.int0([0, 0, 0])
        self.model_factory = model_factory
        self.bbox = bbox
        self.innerBbox = innerBbox
        self.img = self.bbox.crop(img)

        high_contrast_img = cv2.convertScaleAbs(self.img, alpha=1.5)
        high_contrast_img = torch.as_tensor(high_contrast_img)
        high_contrast_img = torch.permute(high_contrast_img, (2, 0, 1))
        self.high_contrast_img = high_contrast_img

    def preprocess(self, detect_font=True):
        self.extract_mask()
        if detect_font:
            self.detect_font()

    def extract_mask(self):
        with torch.no_grad():
            torch_mask = self.model_factory.extract_mask(torch.unsqueeze(self.high_contrast_img, dim=0))[0]
        torch_mask = torch_mask.detach()
        mask = torch.permute(torch_mask, (1, 2, 0)).numpy()
        mask = (mask * 255).astype("uint8")
        _, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_TOZERO)

        self.color = extract_color(self.img, mask)

        torch_mask = mask.astype("float32")/255.
        torch_mask = torch.from_numpy(np.expand_dims(torch_mask, axis=-1))
        self.torch_mask = torch.permute(torch_mask, (2, 0, 1))

        if self.bbox.height() < 30:
            print("large dilate")
            kernel = np.ones((9, 9), np.uint8)
        else:
            kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.resize(mask, (self.img.shape[1], self.img.shape[0]))
        self.mask = mask

    def get_mask_and_bbox(self, img_width, img_height, total_area):
        return self.mask, self.bbox.get()

    def detect_font(self):
        self.font_path = self.model_factory.detect_font(torch.unsqueeze(self.torch_mask, dim=0))[0]
        print(self.text, self.font_path)

    def __str__(self):
        result = f"""

        =====================
        Bbox: {self.bbox}
        Width: {self.bbox.width()}
        Height: {self.bbox.height()}
        Text: {self.text}
        """
        return result

    def __repr__(self):
        return self.__str__()


class TextSwapper:
    def __init__(self, model_factory, img, unified=False, translated_paragraph=None):
        '''
        model_factory: ModelFactory object
        img: cv2 image
        unified: treats all the lines as one unified paragraph
        '''

        self.model_factory = model_factory
        self.img = img
        self.rois = []
        self.translated_paragraph = translated_paragraph
        self.num_sentences = 0
        self.max_line_length = 0
        self.unified = unified

    def detect_text(self, img_path):
        result = self.model_factory.detect_text(img_path)
        if len(result) == 0:
            return False, "No text found"

        temp = self.img.copy()
        paragraph = ""
        all_bboxes = []
        word_rois = []
        for i, item in enumerate(result):
            boxes, text = item

            # add to the paragraph
            paragraph += text.strip() + " "

            all_bboxes.append(boxes)
            boxes = np.int32(boxes)
            x1 = np.min(boxes[:, 0])
            x2 = np.max(boxes[:, 0])
            y1 = np.min(boxes[:, 1])
            y2 = np.max(boxes[:, 1])
            roi = WordRoi(BBox(x1, y1, x2, y2), text)
            word_rois.append(roi)

            # list of four points
            cv2.drawContours(temp, [boxes], -1, (0, 255, 0), 1)

        print("Original paragraph\n", paragraph)
        temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
        if self.translated_paragraph is None:
            self.translated_paragraph = self.model_factory.translate(paragraph)
        print("Translated\n", self.translated_paragraph)

        all_bboxes = np.concatenate(all_bboxes).astype("int32")
        x1 = np.min(all_bboxes[:, 0])
        x2 = np.max(all_bboxes[:, 0])
        y1 = np.min(all_bboxes[:, 1])
        y2 = np.max(all_bboxes[:, 1])
        self.paragraph_bbox = BBox(x1, y1, x2, y2)
        self.segment_sentences(word_rois)

        return True, ""

    def combine_roi(self, id, roi_list):
        text = " ".join([r.text.strip() for r in roi_list])
        x1 = min([r.bbox.x1 for r in roi_list])
        x2 = max([r.bbox.x2 for r in roi_list])
        y1 = min([r.bbox.y1 for r in roi_list])
        y2 = max([r.bbox.y2 for r in roi_list])
        originalBbox = BBox(x1, y1, x2, y2)
        x1, y1, x2, y2 = expand_bbox([x1, y1, x2, y2], self.img.shape)
        bbox = BBox(x1, y1, x2, y2)
        innerBbox = BBox(originalBbox.x1 - bbox.x1,
                         originalBbox.y1 - bbox.y1,
                         originalBbox.x2 - bbox.x1,
                         originalBbox.y2 - bbox.y1)

        return Roi(id, self.img, innerBbox, bbox, text, self.model_factory)

    def segment_sentences(self, word_rois):
        '''
        Count the number of sentence and max number of characters per line
        '''
        if len(word_rois) == 0:
            return

        sentence = [word_rois[0]]
        self.rois = []
        roi_lengths = []
        for roi in word_rois[1:]:
            if roi.bbox.y1 > sentence[-1].bbox.center_point()[1]:
                new_roi = self.combine_roi(len(self.rois), sentence)
                self.rois.append(new_roi)
                roi_lengths.append(len(new_roi.text))
                print(new_roi.text)
                sentence = [roi]
            else:
                sentence.append(roi)

        if len(sentence) > 0:
            new_roi = self.combine_roi(len(self.rois), sentence)
            self.rois.append(new_roi)
            roi_lengths.append(len(new_roi.text))
            print(new_roi.text)

        # check unified and center align
        self.check_unified()
        center_align = self.check_center_align()

        # break paragraph into line with appropriate lengths
        text_distributor = TextDistributor(roi_lengths, self.translated_paragraph)
        print(roi_lengths)
        print(text_distributor.segmented_text)

        for i in range(min(len(text_distributor.segmented_text), len(self.rois))):
            self.rois[i].target_text = text_distributor.segmented_text[i]
            self.rois[i].center_align = center_align

        # preprocess and detect_font
        font_path = None
        if self.unified:
            text_h = 0
            max_roi = max([(len(r.target_text), r) for r in self.rois], key=lambda x: x[0])[1]
            max_roi.preprocess(detect_font=True)
            font_path = max_roi.font_path
            text_h = max_roi.text_h

            for roi in self.rois:
                roi.font_path = font_path
                roi.text_h = text_h
                roi.preprocess(detect_font=False)
        else:
            for roi in self.rois:
                if font_path is None:
                    roi.preprocess(detect_font=True)
                    font_path = roi.font_path
                    continue
                roi.font_path = font_path
                roi.preprocess(detect_font=False)

    def check_unified(self):
        roi_heights = [roi.innerBbox.height() for roi in self.rois]
        for i in range(1, len(roi_heights)):
            a = max(min(roi_heights[i], roi_heights[i-1]), 0.00001)
            b = max(roi_heights[i], roi_heights[i-1])
            print("Unified value", b, a, b/a)
            if b/a > 1.5:
                print("NOT UNIFIED")
                self.unified = False
                return
        print("UNIFIED")
        self.unified = True

    def check_center_align(self):
        if len(self.rois) <= 1:
            return True

        for i in range(1, len(self.rois)):
            a = self.rois[i].bbox
            b = self.rois[i-1].bbox
            if abs(a.x1 - b.x1) > a.height() * 0.5:
                print("CENTER ALIGN")
                return True
        print("LEFT ALIGN")
        return False

    def create_mask(self):
        mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype="uint8")
        total_area = mask.shape[0] * mask.shape[1]
        for roi in self.rois:
            roi_mask, (x1, y1, x2, y2) = roi.get_mask_and_bbox(mask.shape[1], mask.shape[0], total_area)
            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], roi_mask)
        return mask

    def extract_background(self):
        print("Extracting background...")
        mask = self.create_mask()
        res_img = self.model_factory.extract_background([self.img], [mask])
        res_img = res_img[0]
        self.bg_img = res_img
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(img_dir, "bg.png"), res_img)
        print("Done")

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect


def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	M_inverse = cv2.getPerspectiveTransform(dst, rect)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped, M_inverse


def get_roi_from_img(original_img):
    refPts = []
    img = original_img.copy()
    warpedImg = img
    M_inverse = None
    def click_and_crop(event, x, y, flags, param):
        nonlocal refPts, warpedImg, M_inverse
        if event == cv2.EVENT_LBUTTONDOWN:
            refPts.append((x, y))
            cv2.circle(img, (x, y), 3, (255, 255, 0), -1)
        elif event == cv2.EVENT_LBUTTONUP and len(refPts) == 4:
            box = np.int0(refPts)
            cv2.drawContours(img, [box], 0, (0,191,255), 2)

            warpedImg, M_inverse = four_point_transform(original_img.copy(), box)
            refPts = []
            cv2.imshow("warped", warpedImg)


    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    return warpedImg, M_inverse


def static_main():
    model_factory = ModelFactory("./weights", "./fonts")
    img_path = os.path.join(img_dir, "original.png")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st = time()
    text_swapper = TextSwapper(model_factory, img, unified=True,
                               translated_paragraph=None)

    success, mess = text_swapper.detect_text(img_path)
    if not success:
        print(mess)
        return

    text_swapper.extract_background()

def roi_main():
    model_factory = ModelFactory("./weights", "./fonts")
    img_path = os.path.join(img_dir, "original.png")
    full_img = cv2.imread(img_path)

    warpedImg, M_inverse = get_roi_from_img(full_img)
    img_path = os.path.join(img_dir, "warped.png")

    cv2.imwrite(img_path, warpedImg)
    img = warpedImg

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # target_text = "chuyên bán máy tính và phụ tùng linh kiện các thứ"
    target_text = "máy tính PC nguyên chất"
    text_swapper = TextSwapper(model_factory, img, unified=True,
                               translated_paragraph=target_text)

    success, mess = text_swapper.detect_text(img_path)
    if not success:
        print(mess)
        return

    text_swapper.extract_background()
    text_swapper.update_bg()
    result_img = text_swapper.fuse_img()

    result_img_inverse = cv2.warpPerspective(result_img, M_inverse,
                                             (full_img.shape[1], full_img.shape[0]))
    mask = np.ones(result_img.shape).astype("uint8") * 255
    mask_inverse = cv2.warpPerspective(mask, M_inverse,
                                       (full_img.shape[1], full_img.shape[0]))
    mask_inverse = cv2.erode(mask_inverse, np.ones((5, 5), dtype="uint8"), iterations=1)

    mask_inverse = mask_inverse.astype("float32") / 255.
    combine_img = full_img * (1 - mask_inverse) + result_img_inverse * mask_inverse
    cv2.imwrite(os.path.join(img_dir, "combined.png"), combine_img)

if __name__ == "__main__":
    static_main()
    # roi_main()
