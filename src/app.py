import os

import streamlit as st
import cv2
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
from model.resunet import *
import torchvision.transforms as T
from utils import to_Tensor
import io
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_holes, remove_small_objects, label
from skimage.segmentation import watershed
from scipy import ndimage
from math import hypot
import numpy as np
import random
import PIL
from streamlit.runtime.legacy_caching import clear_cache


from src.utils import load_model

from src.utils import *
import matplotlib
matplotlib.use('TkAgg')

import cv2

@st.cache()
def load_model(path='../pre_trained_models', n_features_start=16, n_out=1, fine_tuning=False,
               unfreezed_layers=1, device='cpu', model_to_load='yellow'):
    """Retrieves the trained model and maps it to the CPU by default,
    can also specify GPU here."""
    st.session_state.result = 0
    st.session_state.post_processing = 0
    if model_to_load == 'green':
        path = os.path.join(path, "c-resunet_g.h5")
    elif model_to_load == 'yellow':
        path = os.path.join(path, "c-resunet_y.h5")

    if fine_tuning:
        model = load_model(resume_path=path, device=device, n_features_start=n_features_start, n_out=n_out,
                           fine_tuning=fine_tuning, unfreezed_layers=unfreezed_layers).to(device)
    else:
        model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=n_features_start, n_out=1, c0=True))
        try:
            if device == 'cpu':
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model_state_dict'])
            else:
                model.load_state_dict(torch.load(path)['model_state_dict'])
        except:
            if device == 'cpu':
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            else:
                model.load_state_dict(torch.load(path))

    return model

@st.cache(allow_output_mutation=True)
def predict(model, images, device='cpu', transform=None):
    model.eval()
    preds = []

    if isinstance(images, list):
        for img in images:
            input_tensor = transform(img)[:3, :,:]  # transforming from pil to tensor already have value between 0 and 1
            # (3, h, w) images
            input_batch = input_tensor.unsqueeze(0).to(device)  # batch the image
            # TODO: when come multiple images pack in a list and iter to
            # TODO: transform in tensor or directly transform al the images in the list
            with torch.no_grad():
                preds.append(model(input_batch).detach().cpu())  # (1, h, w) size images # TODO: make for loop whne multiple files
                if device == 'cuda':
                    torch.cuda.empty_cache()
    else:
        input_tensor = transform(images)[:3, :,:]  # transforming from pil to tensor already have value between 0 and 1
        # (3, h, w) images
        input_batch = input_tensor.unsqueeze(0).to(device)  # batch the image
        # TODO: when come multiple images pack in a list and iter to
        # TODO: transform in tensor or directly transform al the images in the list
        with torch.no_grad():
            preds.append(model(input_batch).detach().cpu())  # (1, h, w) size images # TODO: make for loop whne multiple files
            if device == 'cuda':
                torch.cuda.empty_cache()

    return preds
@st.cache(allow_output_mutation=True)
def binarize(preds, th=0.7):

    if isinstance(preds, list):
        for p in preds:
            preds_t = [(np.squeeze(x[0:1, :, :]) > th) for x in preds]
    else:
        preds_t = (np.squeeze(x[0:1, :, :]) > th)

    return preds_t

@st.cache
def make_post_processing(preds, area_threshold=6, min_obj_size=2, max_dist=3, foot=4):
    '''
     preds: array of tensor (ch, h, w)
     targets: array of tensor (ch, h, w)
     return:
     processed_preds: array of tensor (ch, h, w)
     targets: array of tensor (ch, h, w)
     '''

    if len(preds[0].shape) > 2:
        ix = np.argmin(preds[0].shape)
        if ix != 0:
            raise Exception("channels are not on the first dimension \
                            or are more than the spatial dimension")

    # Find object in predicted image
    processed_preds = []
    for p in preds:

        labels_pred, nlabels_pred = ndimage.label(p)
        processed = remove_small_holes(labels_pred, area_threshold=area_threshold, connectivity=1,
                                       in_place=False)
        processed = remove_small_objects(processed, min_size=min_obj_size,
                                         connectivity=1, in_place=False)
        labels_bool = processed.astype(bool)
        distance = ndimage.distance_transform_edt(processed)

        maxi = ndimage.maximum_filter(distance, size=max_dist, mode='constant')
        local_maxi = peak_local_max(np.squeeze(maxi), indices=False, footprint=np.ones((foot, foot)),
                                    exclude_border=False,
                                    labels=np.squeeze(labels_bool))
        local_maxi = remove_small_objects(
            local_maxi, min_size=min_obj_size, connectivity=1, in_place=False)
        markers = ndimage.label(local_maxi)[0]
        labels = watershed(-distance, markers, mask=np.squeeze(labels_bool),
                           compactness=1, watershed_line=True)
        processed_preds.append(labels.astype("uint8")*255)

    return processed_preds


@st.cache(allow_output_mutation=True)
def load_image(uploaded_file):
    print('uploading images')
    st.session_state.result = 0 # TODO: if the new images are a subset of the previous one, keen the st.session_state_result = 1
    st.session_state.post_processing = 0

    if isinstance(uploaded_file, list):
        if len(uploaded_file) == 0:
            filenames_to_read = os.listdir('../images')
            images = []
            for fl in filenames_to_read:
                #st.image('../images/{}'.format(fl))
                image_data = Image.open('../images/{}'.format(fl))
                img_byte_arr = io.BytesIO()
                image_data.save(img_byte_arr, format='PNG')  # TODO: switch to tiff format
                image_data = img_byte_arr.getvalue()
                images.append(Image.open(io.BytesIO(image_data)))
            return images, filenames_to_read
        else:
            images = []
            filenames_to_read = []
            for img in uploaded_file:
                image_data = img.getvalue()
                filenames_to_read.append(img.name)
                images.append(Image.open(io.BytesIO(image_data)))
            return images, filenames_to_read
    ### When accept_multiple file is false:
    else:
        if uploaded_file is not None:
            image_data = uploaded_file.getvalue()
            st.image(image_data)
            return Image.open(io.BytesIO(image_data)), image_data.name
        else:
            st.image('../images/demo.tiff')
            image_data = Image.open('../images/demo.tiff')
            img_byte_arr = io.BytesIO()
            image_data.save(img_byte_arr, format='PNG') # TODO: switch to tiff format
            image_data = img_byte_arr.getvalue()
            return Image.open(io.BytesIO(image_data)), image_data.name


def display_images(images, filenames):
    sorted_images = images
    sorted_filenames = filenames


    if len(sorted_images) > 5:
        item_to_display = 5
    else:
        item_to_display = len(images)

    ncol = st.sidebar.number_input("how many loaded items to display", 1, len(images), item_to_display)
    shuffle = st.sidebar.number_input("display in random order", 0, 1, 0)

    if shuffle:
        zipped = list(zip(images, filenames))
        random.shuffle(zipped)
        images, filenams = zip(*zipped)
        cols = st.columns(ncol)
        idxs = list(range(0, len(images)))
        for i, x in enumerate(cols):
            # x.selectbox(f"Input # {filenames[i]}", idxs, key=i)
            cols[i].image(images[i])
    else:
        cols = st.columns(ncol)
        idxs = list(range(0, len(images)))
        for i, x in enumerate(cols):
            # x.selectbox(f"Input # {filenames[i]}", idxs, key=i)
            cols[i].image(images[i])

def computing_counts(images, preds):
    # extract predicted objects and counts,
    if isinstance(preds, list):
        counts = []
        bboxes_images = []
        for p, i in zip(preds, images):
            i_draw = ImageDraw.Draw(i)

            pred_label, pred_count = ndimage.label(p)
            pred_objs = ndimage.find_objects(pred_label)

            # compute centers of predicted objects
            #pred_centers = []
            for ob in pred_objs:
                #pred_centers.append(((int((ob[0].stop - ob[0].start) / 2) + ob[0].start),
                #                     (int((ob[1].stop - ob[1].start) / 2) + ob[1].start)))
                #cv2.rectangle(i_cv, (ob[1].start, ob[0].start), (ob[1].stop, ob[0].stop), (0, 255, 0), 4)
                i_draw.line( ((ob[1].start -20, ob[0].start -20),
                              (ob[1].stop + 20, ob[0].start-20),
                              (ob[1].stop +20, ob[0].stop+20),
                              (ob[1].start-20, ob[0].stop+20),
                              (ob[1].start-20, ob[0].start-20)), fill="green", width=9)

                #i_draw.rectangle([(ob[1].start, ob[0].start), (ob[1].stop, ob[0].stop)], fill=None, outline='green')

            bboxes_images.append(i)
            counts.append(pred_count)
        return bboxes_images, counts

    else:
        pred_label, pred_count = ndimage.label(preds)
        pred_objs = ndimage.find_objects(pred_label)

        # compute centers of predicted objects
        pred_centers = []
        for ob in pred_objs:
            pred_centers.append(((int((ob[0].stop - ob[0].start) / 2) + ob[0].start),
                                 (int((ob[1].stop - ob[1].start) / 2) + ob[1].start)))

            cv2.rectangle(images, (ob[0].start, ob[0].stop), (ob[1].start, ob[1].stop))

        return images


def main():
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    to_PIL = T.ToPILImage()

    if 'result' not in st.session_state:
        st.session_state.result = 0

    if 'post_processing' not in st.session_state:
        st.session_state.post_processing = 0

    if 'batch_counts' not in st.session_state:
        st.session_state.batch_counts = 0

    #check_model = st.checkbox('select model and training settings')
    uploaded_file = st.file_uploader(label='Pick an image to test', accept_multiple_files = True)
    #if check_model:
    #    model_to_load = st.selectbox('which model to load', np.array(['green', 'yellow']))
    #    model_training_status = st.selectbox('which model to load', np.array(['pre-trained']))
    #else:
    #    model_to_load = 'green'
    #    model_training_status = 'pre-training'

    st.info("Select the model to load:")
    st.markdown(
        """
        <style>
        [data-baseweb="select"] {
            margin-top: -50px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    model_to_load = st.selectbox("", np.array(['green', 'yellow']))
    #st.multiselect('which model to load', np.array([]))

    #cached
    model = load_model(device=device, model_to_load=model_to_load)
    #cached
    images, filenames = load_image(uploaded_file)
    #not chached
    display_images(images, filenames)

    post_processing_title = '<p style="font-family:sans-serif; color:Green; font-size: 16px;">Post-processing parametes</p>'
    st.sidebar.markdown(post_processing_title, unsafe_allow_html=True)

    result = st.button('Make Prediction', key='make_prediction')
    if result:
        st.session_state.result = 1


    if st.session_state.result:
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.6, 0.05)

        preds = predict(model, images, device, transform=to_Tensor)
        preds_th = binarize(preds, th=confidence_threshold)
        preds_to_PIL = [to_PIL(x.int()*255) for x in preds_th]
        preds_to_PIL_converted = []

        images_boxes = []
        for i in images:
            images_boxes.append(i.copy())

        bboxes_pil, counts = computing_counts(images_boxes, preds_th)

        for i, p, c in zip(bboxes_pil, preds_to_PIL, counts):#TODO: make a function
            p = p.convert('L')
            preds_to_PIL_converted.append(p)

            col1, col2 = st.columns(2)
            with col1:
                st.image(i, use_column_width=True)
                st.caption('cells detected without post processing: {}'.format(c))
                #post_processing_title = '<p style="font-family:sans-serif; color:Green; font-size: 24px;">Post-processing parametes</p>'
                #st.sidebar.markdown(post_processing, unsafe_allow_html=True)
            with col2:
                st.image(p, use_column_width=True)

        post_processing = st.button('Post-processing')
        if post_processing:
            st.session_state.post_processing = 1

        #reset = st.sidebar.button('reset value', key='reset_value')
        #if reset:
        #    st.session_state.reset = 1

        if st.session_state.post_processing:
            if model_to_load == 'yellow':
                remove_small_object = st.sidebar.slider("small object size, suggested {}".format(200), 0, 1000, 200, 1)
                area_threshold = st.sidebar.slider("minimum area to keep, suggested {}".format(600), 0, 1000, 600, 1)
                max_dist = st.sidebar.slider("max_dist to define different object, suggested {}".format(300), 0, 100, 30, 1)
                foot = st.sidebar.slider("foot, suggested {}".format(400), 0, 100, 40, 1)
            elif model_to_load == 'green':
                remove_small_object = st.sidebar.slider("small object size to remove, suggested {}".format(2), 0, 100, 2, 1)
                area_threshold = st.sidebar.slider("minimum area to keep, suggested {}".format(6), 0, 100, 6, 1)
                max_dist = st.sidebar.slider("max_dist to define different object, suggested {}".format(3), 0, 100, 3, 1)
                foot = st.sidebar.slider("foot, suggested {}".format(4), 0, 100, 4, 1)

        if st.session_state.post_processing:
            post_processed = make_post_processing(preds_th, area_threshold=area_threshold
                                                  , min_obj_size=remove_small_object , max_dist=max_dist, foot=foot)
            post_processed_to_PIL = [to_PIL(x) for x in post_processed]

            images_boxes_proc = []
            for i in images:
                images_boxes_proc.append(i.copy())
            print('postprocessing', st.session_state.post_processing)
            if st.session_state.post_processing:
                bboxes_pil_proc, counts_proc = computing_counts(images_boxes_proc, post_processed)

                for i, p, pp, c, cp in zip(images, bboxes_pil, bboxes_pil_proc, counts, counts_proc):#TODO: make a function
                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(p, use_column_width=True)
                        st.caption('cells detected without post processing: {}'.format(c))

                    with col2:
                        st.image(pp, use_column_width=True)
                        st.caption('cells detected with post_processing: {}'.format(cp))

    batch_counts = st.button('Run batch analysis TO IMPLEMENT', key='batch_analysis')
    if batch_counts:
        st.session_state.batch_counts = 1
    if st.session_state.batch_counts == 1:
        st.write('TO IMPLEMENT')





if __name__ == '__main__':
    main()
