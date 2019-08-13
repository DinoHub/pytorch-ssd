import os
import cv2
import torch

import numpy as np

if __name__ == '__main__':
    from vision.utils.misc import Timer
    from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
else:
    from .vision.utils.misc import Timer
    from .vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

SSD_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

class SSD(object):
    _defaults = {
        "model_path": os.path.join(SSD_MODEL_DIR, 'models/mb2-ssd-lite-mp-0_686.pth'),
        "classes_path": os.path.join(SSD_MODEL_DIR, 'models/voc-model-labels.txt'),
        "top_k": 20,
        "prob_threshold" : 0.4
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.net = create_mobilenetv2_ssd_lite(len(self.class_names), is_test=True)
        self.net.load(self.model_path)
        self.net.eval()
        self.net.cuda()
        self.predictor = create_mobilenetv2_ssd_lite_predictor(self.net, candidate_size=200, device='cuda')


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def detect_ltwh(self, np_image, classes=None, buffer=0.):
        '''
        detect method

        Params
        ------
        np_image : ndarray, ASSUMED to be read from cv2, so in BGR format

        Returns
        ------
        list of triples ([left, top, width, height], score, predicted_class)

        '''
        dets = []

        image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.predictor.predict(image, self.top_k, self.prob_threshold)

        for i in range(boxes.size(0)):
            predicted_class = self.class_names[labels[i]]
            if classes is not None and predicted_class not in classes:
                continue

            box = boxes[i, :].numpy()
            left, top, right, bottom = box
            width = right-left+1
            height = bottom-top+1
            width_buf = width * buffer
            height_buf = height * buffer
            top = max(0, np.floor(top + 0.5 - height_buf).astype('int32'))
            left = max(0, np.floor(left + 0.5 - width_buf).astype('int32'))
            bottom = min(image.shape[1], np.floor(bottom + 0.5 + height_buf).astype('int32'))
            right = min(image.shape[0], np.floor(right + 0.5 + width_buf).astype('int32'))

            score = probs[i]

            dets.append( ([left, top, width, height], score, predicted_class) )

        return dets

if __name__ == '__main__':
    ssd = SSD()
    # img = cv2.imread('/home/levan/Pictures/auba.jpg')
    # image = Image.fromarray( img )

    image = cv2.imread( '/home/angeugn/Workspace/aicamp/data/5poses/split/val/KoreanHeart/KoreanHeart_36_1120.png' )

    dets = ssd.detect_ltwh(image)
    for ltwh, score, pred_class in dets:
        left,top,width,height = ltwh
        right = int(left + width)
        bottom = int(top + height)
        label = "{}: {:.2f}".format(pred_class, score)
        cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 0), 4)

        cv2.putText(image, label,
                    (left+20, top+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    
    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # boxes, scores, classes = yolo.generate()
    # print(boxes)
    # print(scores)
    # print(classes)