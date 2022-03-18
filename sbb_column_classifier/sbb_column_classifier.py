#!/usr/bin/env python3

__version__ = "1.0"

import logging
import mimetypes
from multiprocessing import Pool
import os
import traceback
import warnings
from contextlib import redirect_stderr

import click
import cv2
import numpy as np
from peewee import *
from more_itertools import peekable

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

with redirect_stderr(open(os.devnull, "w")):
    from keras.models import load_model
    from keras.backend import set_session
    import keras

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")


database = SqliteDatabase(None)  # Defer initialization


class Result(Model):
    image_file = CharField(primary_key=True)
    columns = IntegerField()

    class Meta:
        database = database


def _imread_and_prepare(image_file: str, model_input_shape):
    """Read a single image from a file and prepare it for page prediction

    Must be defined at the top-level so it can be pickled for multiprocessing.
    """
    img_in = cv2.imread(image_file)
    if not img_in:
        return None, None, image_file

    # img = self.otsu_copy(image)
    BLUR_TIMES = 1
    for _ in range(BLUR_TIMES):
        img = cv2.GaussianBlur(img_in, (5, 5), 0)

    # n_classes = model.layers[len(model.layers) - 1].output_shape[3]

    dim = model_input_shape[1:]
    img = img / float(255.0)
    img = _resize_image(img, dim[0], dim[1])

    assert img.shape == dim

    return img, img_in, image_file


# XXX HACK HACK HACK
def _imread_and_prepare_HACK(image_file: str):
    model_input_shape = (None, 448, 448, 3)  # XXX hardcoded shape
    return _imread_and_prepare(image_file, model_input_shape)


def _resize_image(img_in, input_height, input_width):
    """
    Resize image.

    Wrap cv2.resize so that the order of the height and width parameters match .shape's order.

    Must be defined at the top-level so it can be pickled for multiprocessing.
    """
    return cv2.resize(img_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)


class sbb_column_classifier:
    def __init__(self, dir_models):

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

        self.model_page_file = os.path.join(dir_models, "model_page_mixed_best.h5")
        self.model_page = self.our_load_model(self.model_page_file)
        self.model_classifier_file = os.path.join(dir_models, "model_scale_classifier.h5")
        self.model_classifier = self.our_load_model(self.model_classifier_file)

    def our_load_model(self, model_file):
        self.logger.debug("Loading model {}...".format(os.path.basename(model_file)))

        model = load_model(model_file, compile=False)
        self.logger.debug(f"  input_shape:  {model.layers[0].input_shape}")
        self.logger.debug(f"  output_shape: {model.layers[len(model.layers) - 1].output_shape}")

        self.logger.debug("Loading model done.")

        return model

    @staticmethod
    def _crop_image_inside_box(box, img_org_copy):
        # input: coordinates of corner + dimensions
        #        box (x, y, w, h) or (y, x, h, w)
        # returns corner coordinates
        image_box = img_org_copy[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
        return image_box, [box[1], box[1] + box[3], box[0], box[0] + box[2]]

    @staticmethod
    def find_bounding_rect_of_largest_blob(img):
        # Find the largest blob in the prediction - it is our page (and its countour is the border)
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])
            cnt = contours[np.argmax(cnt_size)]

            # bounding rectangle of the largest blob
            x, y, w, h = cv2.boundingRect(cnt)

            # TODO Put 30 into a constant
            # If the prediction is near the img borders, go fully to the border
            if x <= 30:
                w = w + x
                x = 0
            if (img.shape[1] - (x + w)) <= 30:
                w = w + (img.shape[1] - (x + w))

            if y <= 30:
                h = h + y
                y = 0
            if (img.shape[0] - (y + h)) <= 30:
                h = h + (img.shape[0] - (y + h))

            box = [x, y, w, h]
        else:
            # If no contour is found, assume the full image as the page
            box = [0, 0, img.shape[1], img.shape[0]]

        return box

    N_WORKERS = 4
    BATCH_SIZE = 32

    def _crop_page_from_pred(self, pred, img_in):
        seg = np.argmax(pred, axis=2)
        seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

        # XXX Here we seem to resize it (just to resize it smaller again later)
        prediction_true = _resize_image(seg_color, img_in.shape[0], img_in.shape[1])
        prediction_true = prediction_true.astype(np.uint8)

        # XXX Why make an RGB image above and then convert it back to grayscale again?
        img_page_prediction = prediction_true
        img_gray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 0, 255, 0)
        thresh = cv2.dilate(thresh, self.kernel, iterations=3)

        box = self.find_bounding_rect_of_largest_blob(thresh)
        cropped_page, page_coord = self._crop_image_inside_box(box, img_in)

        return cropped_page

    def crop_pages_batchwise(self, image_files):
        """Crop pages, batch for batch"""
        batch = []
        # X = np.empty((self.batch_size, *dim))
        with Pool(self.N_WORKERS) as pool:
            prepared_images = peekable(pool.imap(_imread_and_prepare_HACK, image_files))
            for x, img_in, image_file in prepared_images:
                if not x:
                    self.logger.error(f"Error reading {image_file}")
                    continue

                batch.append([x, img_in, image_file])

                # We have either a full batch or the last batch (= peekable iterator is exhausted):
                if len(batch) >= self.BATCH_SIZE or not prepared_images:
                    X = np.stack((x for x, _, _ in batch), axis=0)
                    pred_batch = self.model_page.predict(X)

                    # TODO This doesn't run parallelized
                    cropped_pages = []
                    for label_p_pred, (img_in, image_file2) in zip(pred_batch, ((img_in, image_file2) for _, img_in, image_file2 in batch)):
                        cropped_page = self._crop_page_from_pred(label_p_pred, img_in)
                        cropped_pages.append((cropped_page, image_file2))
                    batch = []
                    yield cropped_pages

    def number_of_columns(self, image_files):
        for cropped_pages_batch in self.crop_pages_batchwise(image_files):
            batch = []
            # TODO This doesn't run parallelized
            for cropped_page, image_file in cropped_pages_batch:
                # TODO ... just to resize it down again
                cropped_page = cropped_page / 255.0
                cropped_page = cv2.resize(cropped_page, (448, 448), interpolation=cv2.INTER_NEAREST)  # XXX hardcoded shape
                batch.append((cropped_page, image_file))

            X = np.stack((x for x, _ in batch), axis=0)
            label_p_pred = self.model_classifier.predict(X)
            num_col_batch = np.argmax(label_p_pred, axis=1) + 1
            self.logger.debug(f"Batch of {len(batch)} done.")
            yield from zip(num_col_batch, (image_file for _, image_file in batch))


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
    cl = sbb_column_classifier(model)

    def is_image(fn):
        guessed_mimetype = mimetypes.guess_type(fn)[0]
        return guessed_mimetype is not None and guessed_mimetype.startswith("image/")

    def already_done(fn):
        return db_out and Result.get_or_none(Result.image_file == fn)

    def process_walk(i, explicitly_given=False):
        if os.path.isdir(i):
            root = i
            # Using os.scandir() here for better performance over os.walk()
            with os.scandir(root) as it:
                for entry in it:
                    yield from process_walk(os.path.join(root, entry.name))
        elif os.path.isfile(i):
            if explicitly_given or is_image(i):
                if already_done(i):
                    cl.logger.debug("Skipping {!r}, it is already done.".format(i))
                else:
                    yield i

    def process_walk_outer(images):
        for i in images:
            yield from process_walk(i, explicitly_given=True)

    if db_out:
        database.init(db_out)
        database.create_tables([Result])

    for number_of_columns, image_file in cl.number_of_columns(process_walk_outer(images)):
        print("{!r},{}".format(image_file, number_of_columns))
        if db_out:
            r = Result.create(image_file=image_file, columns=number_of_columns)

    # TODO
    # try:
    #    cl.run(image_file)
    # except Exception:
    #    print(traceback.format_exc())


if __name__ == "__main__":
    main()
