from skimage.feature import peak_local_max
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as npd
import torch
import torchvision.transforms as T

to_Tensor = T.Compose([T.ToTensor(),
                       #T.Lambda(lambda x: x * 1. / 255)
                       ])


def model_inference(data_loader, model, device = 'cpu'):
    model.eval()
    preds = []
    targets = []
    images = []

    for ix, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            print("batch {} on {}".format(ix, len(data_loader)))
            results = model(x.to(device)).cpu().detach()
            preds.extend(results)
            images.extend(x)
            targets.extend(y)
            torch.cuda.empty_cache()
    return images, targets, preds

# MOVE TO Evlauation utils
def post_processing(preds, targets, th=0.3, min_obj_size=2,
                    foot=4, area_threshold=6, max_dist=3):
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
        preds_t = [(np.squeeze(x[0:1, :, :]) > th) for x in preds]

    if len(targets[0].shape) > 2:
        ix = np.argmin(targets[0].shape)
        if ix != 0:
            raise Exception("channels are not on the first dimension \
                            or are more than the spatial dimension")
        targets = [np.squeeze(x[0:1, :, :]) for x in targets]

    processed_preds = []
    for p, t in zip(preds_t, targets):
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

    return processed_preds, targets

def annotation(
    images, detections, confidence_threshold=0.9
):
    # loop over the detections
    (h, w) = images.shape[:2]
    labels = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
            labels.append(label)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
            )
    return image, labels

def post_processing(thresh_image, area_threshold=600, min_obj_size=200, max_dist=30, foot=40):

    # Find object in predicted image
    labels_pred, nlabels_pred = ndimage.label(thresh_image)
    processed = remove_small_holes(labels_pred, area_threshold=area_threshold, connectivity=1,
                                   in_place=False)
    processed = remove_small_objects(
        processed, min_size=min_obj_size, connectivity=1, in_place=False)
    labels_bool = processed.astype(bool)

    distance = ndimage.distance_transform_edt(processed)

    maxi = ndimage.maximum_filter(distance, size=max_dist, mode='constant')
    local_maxi = peak_local_max(maxi, indices=False, footprint=np.ones((foot, foot)),
                                exclude_border=False,
                                labels=labels_bool)

    local_maxi = remove_small_objects(
        local_maxi, min_size=25, connectivity=1, in_place=False)
    markers = ndimage.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=labels_bool,
                       compactness=1, watershed_line=True)

    return(labels.astype("uint8")*255)
