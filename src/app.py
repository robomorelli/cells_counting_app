import os

import streamlit as st
import cv2
from PIL import Image
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

import cv2


@st.cache()
def predict(model, images, device='cpu', transform=None, th=0.3):
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

    preds_t = [(np.squeeze(x[0:1, :, :]) > th) for x in preds]
    return preds_t


@st.cache
def post_processing(preds, area_threshold=600, min_obj_size=200, max_dist=30, foot=40):

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

@st.cache()
def load_model(path='../pre_trained_models/c-resunet_y.h5', n_features_start=16, n_out=1, fine_tuning=False,
               unfreezed_layers=1, device='cpu'):
    """Retrieves the trained model and maps it to the CPU by default,
    can also specify GPU here."""

    if fine_tuning:
        model = load_model(resume_path=path, device=device, n_features_start=n_features_start, n_out=n_out,
                           fine_tuning=fine_tuning, unfreezed_layers=unfreezed_layers).to(device)
    else:
        model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=n_features_start, n_out=1, c0=True)).to(device)
        try:
            model.load_state_dict(torch.load(path))
        except:
            model.load_state_dict(torch.load(path)['model_state_dict'])
    return model



def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test', accept_multiple_files = True)
    if isinstance(uploaded_file, list):
        if len(uploaded_file) == 0:
            file_to_read = os.listdir('../images')
            images = []
            for fl in file_to_read:
                st.image('../images/{}'.format(fl))
                image_data = Image.open('../images/{}'.format(fl))
                img_byte_arr = io.BytesIO()
                image_data.save(img_byte_arr, format='PNG')  # TODO: switch to tiff format
                image_data = img_byte_arr.getvalue()
                images.append(Image.open(io.BytesIO(image_data)))
            return images
        else:
            images = []
            for img in uploaded_file:
                image_data = img.getvalue()
                images.append(Image.open(io.BytesIO(image_data)))
            st.image(image_data)
            return Image.open(io.BytesIO(image_data))
    else:
        if uploaded_file is not None:
            image_data = uploaded_file.getvalue()
            st.image(image_data)
            return Image.open(io.BytesIO(image_data))
        else:
            st.image('../images/demo.tiff')
            image_data = Image.open('../images/demo.tiff')
            img_byte_arr = io.BytesIO()
            image_data.save(img_byte_arr, format='PNG') # TODO: switch to tiff format
            image_data = img_byte_arr.getvalue()
            return Image.open(io.BytesIO(image_data))


def main():
    st.title('Pretrained model demo')
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    model = load_model(device=device)
    image = load_image()

    result = st.button('Run on image')
        #result = st.button('Run on image')
    st.write('Computing results...')
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.2, 0.05)

    preds = predict(model, image, device, transform=to_Tensor, th=confidence_threshold)
    to_PIL = T.ToPILImage()
    preds = [to_PIL(x.int()*255) for x in preds]

    for p in preds:
        p = p.convert('L')
        st.image(p, caption=f"Thresholded images", use_column_width=True)

    postprocessing = st.button('Post-processing')
    st.write('making post-processing.')

    post_processed = post_processing(preds, area_threshold=600, min_obj_size=200, max_dist=30, foot=40)
    post_processed = [to_PIL(x) for x in post_processed]

    for p in post_processed:
        p = p.convert('L')
        st.image(p, caption=f"Thresholded images", use_column_width=True)



if __name__ == '__main__':
    main()
