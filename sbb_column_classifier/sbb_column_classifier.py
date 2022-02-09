#! /usr/bin/env python3

__version__ = '1.0'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np

from contextlib import redirect_stderr
with redirect_stderr(open(os.devnull, "w")):
    from keras.models import load_model
    from keras.backend import set_session

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import warnings
import click
import json
import logging


warnings.filterwarnings('ignore')


class sbb_column_classifier:
    def __init__(self, dir_models, json):

        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('sbb_column_classifier')
        self.logger.setLevel(logging.DEBUG)


        self.kernel = np.ones((5, 5), np.uint8)


        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # XXX config is unused...
        session = tf.InteractiveSession()
        set_session(session)

        self.model_classifier_file = os.path.join(dir_models, "model_scale_classifier.h5")
        self.model_classifier = self.start_new_session_and_model(self.model_classifier_file)
        self.model_page_file = os.path.join(dir_models, "model_page_mixed_best.h5")
        self.model_page = self.start_new_session_and_model(self.model_page_file)

        self.json = json


    def resize_image(self, img_in, input_height, input_width):
        return cv2.resize(img_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)

    def start_new_session_and_model(self, model_dir):
        # TODO should be renamed
        self.logger.debug("Loading model {}...".format(os.path.basename(model_dir)))
        model = load_model(model_dir, compile=False)
        self.logger.debug("Loading model done.")

        return model

    def do_prediction(self,patches,img,marginal_of_patch_percent=0.1):
        model = self.model_page

        img_height_model = model.layers[len(model.layers) - 1].output_shape[1]
        img_width_model = model.layers[len(model.layers) - 1].output_shape[2]
        #n_classes = model.layers[len(model.layers) - 1].output_shape[3]

        if patches:
            if img.shape[0]<img_height_model:
                img=self.resize_image(img,img_height_model,img.shape[1])

            if img.shape[1]<img_width_model:
                img=self.resize_image(img,img.shape[0],img_width_model)

            #print(img_height_model,img_width_model)
            #margin = int(0.2 * img_width_model)
            margin = int(marginal_of_patch_percent * img_height_model)

            width_mid = img_width_model - 2 * margin
            height_mid = img_height_model - 2 * margin


            img = img / float(255.0)
            #print(sys.getsizeof(img))
            #print(np.max(img))

            img=img.astype(np.float16)

            #print(sys.getsizeof(img))

            img_h = img.shape[0]
            img_w = img.shape[1]

            prediction_true = np.zeros((img_h, img_w, 3))
            mask_true = np.zeros((img_h, img_w))
            nxf = img_w / float(width_mid)
            nyf = img_h / float(height_mid)

            if nxf > int(nxf):
                nxf = int(nxf) + 1
            else:
                nxf = int(nxf)

            if nyf > int(nyf):
                nyf = int(nyf) + 1
            else:
                nyf = int(nyf)

            for i in range(nxf):
                for j in range(nyf):

                    if i == 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model
                    elif i > 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model

                    if j == 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model
                    elif j > 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model

                    if index_x_u > img_w:
                        index_x_u = img_w
                        index_x_d = img_w - img_width_model
                    if index_y_u > img_h:
                        index_y_u = img_h
                        index_y_d = img_h - img_height_model



                    img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]

                    label_p_pred = model.predict(
                        img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]))

                    seg = np.argmax(label_p_pred, axis=3)[0]

                    seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

                    if i==0 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                        :] = seg_color

                    elif i==nxf-1 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0,
                        :] = seg_color

                    elif i==0 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin,
                        :] = seg_color

                    elif i==nxf-1 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                        :] = seg_color

                    elif i==0 and j!=0 and j!=nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                        :] = seg_color

                    elif i==nxf-1 and j!=0 and j!=nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                        :] = seg_color

                    elif i!=0 and i!=nxf-1 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                        :] = seg_color

                    elif i!=0 and i!=nxf-1 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin,
                        :] = seg_color

                    else:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                        :] = seg_color

            prediction_true = prediction_true.astype(np.uint8)

        if not patches:
            img_h_page=img.shape[0]
            img_w_page=img.shape[1]
            img = img /float( 255.0)
            img = self.resize_image(img, img_height_model, img_width_model)

            label_p_pred = model.predict(
                img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))

            seg = np.argmax(label_p_pred, axis=3)[0]
            seg_color =np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = self.resize_image(seg_color, img_h_page, img_w_page)
            prediction_true = prediction_true.astype(np.uint8)

        return prediction_true

    def crop_image_inside_box(self, box, img_org_copy):
        image_box = img_org_copy[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        return image_box, [box[1], box[1] + box[3], box[0], box[0] + box[2]]

    def extract_number_of_columns(self, image_page):
        patches = False

        img_in = image_page / 255.0
        img_in = cv2.resize(img_in, (448, 448), interpolation=cv2.INTER_NEAREST)
        img_in = img_in.reshape(1, 448, 448, 3)
        label_p_pred = self.model_classifier.predict(img_in)
        num_col = np.argmax(label_p_pred[0]) + 1

        return num_col

    def extract_page(self, image_dir):
        patches = False

        image = cv2.imread(image_dir)

        ###img = self.otsu_copy(image)
        for ii in range(1):  # XXX ???
            img = cv2.GaussianBlur(image, (5, 5), 0)

        img_page_prediction = self.do_prediction(patches, img)

        imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        thresh = cv2.dilate(thresh, self.kernel, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])

        cnt = contours[np.argmax(cnt_size)]

        x, y, w, h = cv2.boundingRect(cnt)


        if x<=30:
            w=w+x
            x=0
        if (image.shape[1]-(x+w) )<=30:
            w=w+(image.shape[1]-(x+w) )

        if y<=30:
            h=h+y
            y=0
        if (image.shape[0]-(y+h) )<=30:
            h=h+(image.shape[0]-(y+h) )



        box = [x, y, w, h]

        croped_page, page_coord = self.crop_image_inside_box(box, image)

        self.cont_page=[]
        self.cont_page.append( np.array( [ [ page_coord[2] , page_coord[0] ] ,
                                                    [ page_coord[3] , page_coord[0] ] ,
                                                    [ page_coord[3] , page_coord[1] ] ,
                                                [ page_coord[2] , page_coord[1] ]] ) )

        return croped_page, page_coord

    def run(self, image_dir):
        self.logger.debug("Running for {}...".format(image_dir))
        image_page, _ = self.extract_page(image_dir)
        number_of_columns = int(self.extract_number_of_columns(image_page))

        if self.json:
            print(json.dumps({
                "image_file": image_dir,
                "columns": number_of_columns
            }, indent=4))
        else:
            if number_of_columns==1:
                print('The document has {} column!'.format(number_of_columns))
            else:
                print('The document has {} columns!'.format(number_of_columns))
        self.logger.debug("Run done.")


@click.command()
@click.option('--model', '-m', help='Directory of models (page extractor and classifier)', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--json',        help='Format output as JSON', is_flag=True, default=False)
@click.argument('images', required=True, type=click.Path(exists=True, dir_okay=False), nargs=-1)
def main(model, json, images):
    """
    Determine the number of columns in the document image IMAGES.

    Input document images should be in RGB.
    """

    x = sbb_column_classifier(model, json)
    for image in images:
        x.run(image)


if __name__ == "__main__":
    main()
