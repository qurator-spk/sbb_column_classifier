#! /usr/bin/env python3

__version__ = "1.0"

import logging
import mimetypes
import os
import traceback
import warnings
from contextlib import redirect_stderr

import click
import cv2
import numpy as np
from peewee import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

with redirect_stderr(open(os.devnull, "w")):
    from keras.models import load_model
    from keras.backend import set_session

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")


database = SqliteDatabase(None)  # Defer initialization


class Result(Model):
    image_file = CharField(primary_key=True)
    columns = IntegerField()

    class Meta:
        database = database


class sbb_column_classifier:
    def __init__(self, dir_models, db_out):

        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger("sbb_column_classifier")
        self.logger.setLevel(logging.DEBUG)
        logging.getLogger("peewee").setLevel(logging.INFO)

        self.kernel = np.ones((5, 5), np.uint8)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.InteractiveSession(config=config)
        set_session(session)

        self.model_classifier_file = os.path.join(dir_models, "model_scale_classifier.h5")
        self.model_classifier = self.our_load_model(self.model_classifier_file)
        self.model_page_file = os.path.join(dir_models, "model_page_mixed_best.h5")
        self.model_page = self.our_load_model(self.model_page_file)

        self.db_out = db_out
        if self.db_out:
            database.init(self.db_out)
            database.create_tables([Result])

    def resize_image(self, img_in, input_height, input_width):
        """
        Resize image.

        Wrap cv2.resize so that the order of the height and width parameters match .shape's order.
        """
        return cv2.resize(img_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)

    def our_load_model(self, model_file):
        self.logger.debug("Loading model {}...".format(os.path.basename(model_file)))
        model = load_model(model_file, compile=False)
        self.logger.debug("Loading model done.")

        return model

    def do_prediction(self, img):
        model = self.model_page

        img_height_model = model.layers[len(model.layers) - 1].output_shape[1]
        img_width_model = model.layers[len(model.layers) - 1].output_shape[2]
        # n_classes = model.layers[len(model.layers) - 1].output_shape[3]

        img_h_page = img.shape[0]
        img_w_page = img.shape[1]
        img = img / float(255.0)
        img = self.resize_image(img, img_height_model, img_width_model)

        label_p_pred = model.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))

        seg = np.argmax(label_p_pred, axis=3)[0]
        seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
        prediction_true = self.resize_image(seg_color, img_h_page, img_w_page)
        prediction_true = prediction_true.astype(np.uint8)

        return prediction_true

    def crop_image_inside_box(self, box, img_org_copy):
        image_box = img_org_copy[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
        return image_box, [box[1], box[1] + box[3], box[0], box[0] + box[2]]

    def extract_page(self, image_file):
        """Determine page border and extract the actual page."""

        image = cv2.imread(image_file)

        ###img = self.otsu_copy(image)
        BLUR_TIMES = 1
        for _ in range(BLUR_TIMES):
            img = cv2.GaussianBlur(image, (5, 5), 0)

        img_page_prediction = self.do_prediction(img)

        img_gray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 0, 255, 0)

        thresh = cv2.dilate(thresh, self.kernel, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        if len(contours) > 0:
            cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])
            cnt = contours[np.argmax(cnt_size)]

            x, y, w, h = cv2.boundingRect(cnt)

            if x <= 30:
                w = w + x
                x = 0
            if (image.shape[1] - (x + w)) <= 30:
                w = w + (image.shape[1] - (x + w))

            if y <= 30:
                h = h + y
                y = 0
            if (image.shape[0] - (y + h)) <= 30:
                h = h + (image.shape[0] - (y + h))

            box = [x, y, w, h]
        else:
            box = [0, 0, img.shape[1], img.shape[0]]

        # Crop page from image
        cropped_page, page_coord = self.crop_image_inside_box(box, image)

        return cropped_page, page_coord

    def extract_number_of_columns(self, image_page: cv2.Mat) -> int:
        """Determine the number of columns of the given page"""
        img_in = image_page / 255.0
        img_in = cv2.resize(img_in, (448, 448), interpolation=cv2.INTER_NEAREST)
        img_in = img_in.reshape(1, 448, 448, 3)
        label_p_pred = self.model_classifier.predict(img_in)
        num_col = np.argmax(label_p_pred[0]) + 1

        return num_col

    def run(self, image_file):
        self.logger.debug("Running for {}...".format(image_file))
        image_page, _ = self.extract_page(image_file)
        number_of_columns = int(self.extract_number_of_columns(image_page))

        if self.db_out:
            r = Result.create(image_file=image_file, columns=number_of_columns)
        print("The document image {!r} has {} {}!".format(image_file, number_of_columns, "column" if number_of_columns == 1 else "columns"))
        self.logger.debug("Run done.")


@click.command()
@click.option(
    "--model",
    "-m",
    help="Directory of models (page extractor and classifier)",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("--db-out", help="Write output as SQLite3 db", type=click.Path(exists=False, dir_okay=False))
@click.argument("images", required=True, type=click.Path(exists=True, dir_okay=True), nargs=-1)
def main(model, db_out, images):
    """
    Determine the number of columns in the document image IMAGES.

    Input document images should be in RGB. If a directory is given as IMAGES,
    we will process any image in the directory and its subdirectories.
    """
    cl = sbb_column_classifier(model, db_out)

    def process(image_file):
        if not db_out or not Result.get_or_none(Result.image_file == image_file):
            try:
                cl.run(image_file)
            except Exception:
                print(traceback.format_exc())
        else:
            cl.logger.debug("Skipping {!r}, it is already done.".format(image_file))

    def is_image(fn):
        guessed_mimetype = mimetypes.guess_type(fn)[0]
        return guessed_mimetype is not None and guessed_mimetype.startswith("image/")

    def process_walk(i, explicitly_given=False):
        if os.path.isdir(i):
            root = i
            # Using os.scandir() here for better performance over os.walk()
            with os.scandir(root) as it:
                for entry in it:
                    process_walk(os.path.join(root, entry.name))
        elif os.path.isfile(i):
            if explicitly_given or is_image(i):
                process(i)

    for i in images:
        process_walk(i, explicitly_given=True)



if __name__ == "__main__":
    main()
