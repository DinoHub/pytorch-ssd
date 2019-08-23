import numpy as np
import logging
import pathlib
import cv2
import os
from PIL import Image


class PP_noBG_Dataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for DH's PP ship data. The huge one with ships and nothing but ships.
        Args:
            root: the root of the PP ship dataset that DH collected, the directory contains the following sub-directories:
                images, labels.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = os.path.join(self.root, "pp.val") 
        else:
            image_sets_file = os.path.join(self.root, "pp.train")
        self.ids = PP_noBG_Dataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = os.path.join(self.root, "pp.names")

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                classes = [c.strip() for c in infile.readlines()]

            # prepend BACKGROUND as first class
            # classes.insert(0, 'BACKGROUND')
            classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("PP Labels read from file: " + str(self.class_names))
        else:
            logging.info("No labels file, using default PP classes (just ship lol).")
            self.class_names = ('ship')
            # self.class_names = ('BACKGROUND','ship')


        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        print(self.class_dict)

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)

        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if boxes is not None and len(boxes) > 0:
            if self.transform:
                image, boxes, labels = self.transform(image, boxes, labels)
            if self.target_transform:
                boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _extract_image_id(image_file_path):
        return image_file_path.rstrip().split('/')[-1]

    # Just image ids, extension removed
    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for image_file_path in f:
                ids.append( PP_noBG_Dataset._extract_image_id( image_file_path ) )
        return ids

    def _get_image_filepath(self, image_id):
        return os.path.join(self.root, 'images/{}'.format( image_id ))

    def _get_annotation_filepath(self, image_id):
        im_id, _ = image_id.split('.')
        return os.path.join(self.root, 'labels/{}.txt'.format( im_id ))

    def _get_annotation(self, image_id):
        image_file = self._get_image_filepath( image_id )
        im_width, im_height = Image.open(image_file).size

        annotation_file = self._get_annotation_filepath( image_id )
        if not os.path.exists( annotation_file ):
            print('{} has no annotations'.format(image_id))
            return (np.array([ [0,0,im_width,im_height] ], dtype=np.float32),
                    np.array([ 0 ], dtype=np.int64),
                    np.array([ 0 ], dtype=np.uint8))

        boxes = []
        labels = []
        is_difficult = []

        # annotation_file = self._get_annotation_filepath( image_id )
        with open(annotation_file, 'r') as annf:
            lines = [ann.strip() for ann in annf.readlines()]
            for line in lines:
                label, x, y, w, h = [float(tok) for tok in line.split(' ')]
                labels.append( int(label) )
                # labels.append( int(label) + 1 )

                xmin = (x - w/2) * im_width
                xmax = (x + w/2) * im_width
                ymin = (y - h/2) * im_height
                ymax = (y + h/2) * im_height
                boxes.append( [xmin, ymin, xmax, ymax] )

                is_difficult.append(0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self._get_image_filepath( image_id )
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


if __name__ == '__main__':

    def unit_test1():
        target_image = '/media/dh/DATA4TB/Datasets/coco/images/val2014/COCO_val2014_000000000042.jpg'
        from PIL import Image
        import cv2
        import time
        start = time.time()
        iters = 1000
        for _ in range(iters):
            cv2_h, cv2_w = cv2.imread( target_image ).shape[:2]
        cv2_elapsed = time.time() - start

        print('h={},w={} for cv2'.format(cv2_h, cv2_w))
        print('avg for cv2: {}'.format( cv2_elapsed / iters ))

        start = time.time()
        for _ in range(iters):
            im = Image.open(target_image)
            pil_w, pil_h = im.size
        pil_elapsed = time.time() - start

        print('h={},w={} for PIL'.format(pil_h, pil_w))
        print('avg for PIL: {}'.format( pil_elapsed / iters ))

    def unit_test2():
        root = '/media/dh/DATA4TB/Datasets/pp_modir'
        image_id = '2008_003951.jpg'
        dataset = PP_noBG_Dataset( root )

        b,l,d = dataset._get_annotation( image_id )
        print(b)
        print(l)

    unit_test2()
