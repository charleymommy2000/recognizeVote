import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.filters import threshold_local
import uuid
import os
import json
from azure.storage.blob import BlobServiceClient
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
from io import BytesIO
import urllib.request
from urllib.request import urlopen

import logging

import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    image_bytes = get_image_bytes_from_request(req)
    message = get_vote_from_document(image_bytes)

    return func.HttpResponse(message, status_code=200)

def get_image_bytes_from_request(req):
    image_url = req.params.get('url')
    if not image_url:
        image_bytes = req.get_body()
    else:
        image_bytes = urlopen(image_url).read()
    return image_bytes

def convert_img_bytes_to_cv2_img(image_bytes):

    img_nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_nparr, cv2.IMREAD_COLOR)
    return img

def prepare_json_message(vote,isSigned,isValid,origFileUrl,voteRecognizedFileUrl,lat,lng,date_time):
    message = {}
    message['vote'] = vote
    message['signed'] = isSigned
    message['valid'] = isValid
    message['origFileUrl'] = origFileUrl
    message['voteRecognizedFileUrl'] = voteRecognizedFileUrl
    message['latitude'] = lat
    message['longitude'] = lng
    message['dateTimeOriginal'] = date_time
    return json.dumps(message)

class ImageMetaData(object):
    '''
    Extract the exif data from any image. Data includes GPS coordinates, 
    Focal Length, Manufacture, and more.
    '''
    exif_data = None
    image = None

    def __init__(self, img_data):
        self.image = Image.open(io.BytesIO(img_data))
        #print(self.image._getexif())
        self.get_exif_data()
        super(ImageMetaData, self).__init__()

    def get_exif_data(self):
        """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
        exif_data = {}
        info = self.image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]

                    exif_data[decoded] = gps_data
                else:
                    exif_data[decoded] = value
        self.exif_data = exif_data
        return exif_data

    def get_if_exist(self, data, key):
        if key in data:
            return data[key]
        return None

    def convert_to_degress(self, value):

        """Helper function to convert the GPS coordinates 
        stored in the EXIF to degress in float format"""
        d0 = value[0][0]
        d1 = value[0][1]
        d = float(d0) / float(d1)

        m0 = value[1][0]
        m1 = value[1][1]
        m = float(m0) / float(m1)

        s0 = value[2][0]
        s1 = value[2][1]
        s = float(s0) / float(s1)

        return d + (m / 60.0) + (s / 3600.0)

    def get_lat_lng(self):
        """Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)"""
        lat = None
        lng = None
        exif_data = self.get_exif_data()
        #print(exif_data)
        if "GPSInfo" in exif_data:      
            gps_info = exif_data["GPSInfo"]
            gps_latitude = self.get_if_exist(gps_info, "GPSLatitude")
            gps_latitude_ref = self.get_if_exist(gps_info, 'GPSLatitudeRef')
            gps_longitude = self.get_if_exist(gps_info, 'GPSLongitude')
            gps_longitude_ref = self.get_if_exist(gps_info, 'GPSLongitudeRef')
            if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
                lat = self.convert_to_degress(gps_latitude)
                if gps_latitude_ref != "N":                     
                    lat = 0 - lat
                lng = self.convert_to_degress(gps_longitude)
                if gps_longitude_ref != "E":
                    lng = 0 - lng
        return lat, lng
    
    def get_date_time_original(self):
        """Returns the date and time when photo was taken, if available, from the provided exif_data (obtained through get_exif_data above)"""
        exif_data = self.get_exif_data()
        if "DateTimeOriginal" in exif_data:
            date_and_time = self.exif_data['DateTimeOriginal']
            return date_and_time     

def get_meta_data_from_image(img_bytes):
    lat,lng = get_img_location(img_bytes)
    date_time = get_img_datetime(img_bytes)
    return lat,lng,date_time

def get_img_location(img_bytes):
    meta_data =  ImageMetaData(img_bytes)
    return meta_data.get_lat_lng()

def get_img_datetime(img_bytes):
    meta_data =  ImageMetaData(img_bytes)
    return meta_data.get_date_time_original()    

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def warp_image(image):
    #ratio = image.shape[0] / 500.0
    ratio = 1
    orig = image.copy()
    #image = imutils.resize(image, height = 500)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
            
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)   
    return warped

def normalize(im):
    """Converts `im` to black and white.

    Applying a threshold to a grayscale image will make every pixel either
    fully black or fully white. Before doing so, a common technique is to
    get rid of noise (or super high frequency color change) by blurring the
    grayscale image with a Gaussian filter."""
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Filter the grayscale image with a 3x3 kernel
    blurred = cv2.GaussianBlur(im_gray, (3, 3), 0)

    # Applies a Gaussian adaptive thresholding. In practice, adaptive thresholding
    # seems to work better than appling a single, global threshold to the image.
    # This is particularly important if there could be shadows or non-uniform
    # lighting on the answer sheet. In those scenarios, using a global thresholding
    # technique might yield paricularly bad results.
    # The choice of the parameters blockSize = 77 and C = 10 is as much as an art
    # as a science and domain-dependand.
    # In practice, you might want to try different  values for your specific answer
    # sheet.
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 77, 10)
    
    return thresh

def get_contours(image_gray):
    cnts = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def isSquare(contour):
    threshold_max_area = 600
    threshold_min_area = 500
    aspect_ratio_min = 0.8
    aspect_ratio_max = 1.1
    
    #Identify the shape using area, threshold, aspect ratio, contours closed/open 
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
    x,y,w,h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    area = cv2.contourArea(contour) 
    
    return ((len(approx) == 4) and (area > threshold_min_area and area < threshold_max_area) and (aspect_ratio >= aspect_ratio_min and aspect_ratio <= aspect_ratio_max))

def get_square_info(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
    x,y,w,h = cv2.boundingRect(approx)
    return x,y,w,h

def get_image_size(im):
    height = im.shape[0]
    width = im.shape[1]
    return height, width

def get_dark_level(checkbox):
    n_white_pix = np.sum(checkbox == 255)
    n_black_pix = np.sum(checkbox == 0)
    if n_white_pix == 0 :
        dark_level = 1
    else :
        dark_level = n_black_pix/n_white_pix
    return dark_level

def get_cropped_image(im,x,y,w,h,i):
    im_crop = im[y + i:y + h - i,x + i:x + w - i] 
    return im_crop

def crop_image(im,crop,cropRight,cropBottom):
    if not crop:
        return im
    else:
        if cropRight:
            if cropBottom:
                return get_right_bottom_image_part(im)
            else:
                return get_right_upper_image_part(im)
        else:
            if cropBottom:
                return get_left_bottom_image_part(im)
            else:
                return get_left_upper_image_part(im)

def get_square_insights(cb_im):
    contours = cv2.findContours(cb_im, cv2.RETR_CCOMP,1)
    cntLen = 1
    ct = 0 #number of contours
    for cnt in contours:
        if len(cnt) > cntLen: #eliminate the noises
            ct += 1
                    
    dark_level = get_dark_level(cb_im)     
    return ct, dark_level 

def draw_contour(contour,im):
    image = im.copy()
    cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
    plt.imshow(image)
    plt.show()

def get_right_bottom_image_part(im):
    h,w = get_image_size(im)
    x = int(w/2)
    y = int(h/2)
    w = int(w/2)
    h = int(h/2)
    return get_cropped_image(im, x,y,w,h,0)

def get_right_upper_image_part(im):
    h,w = get_image_size(im)
    x = int(w/2)
    y = int(0)
    w = int(w/2)
    h = int(h/2)
    return get_cropped_image(im, x,y,w,h,0)

def get_left_bottom_image_part(im):
    h,w = get_image_size(im)
    x = int(0)
    y = int(h/2)
    w = int(w/2)
    h = int(h/2)
    return get_cropped_image(im, x,y,w,h,0)

def get_left_upper_image_part(im):
    h,w = get_image_size(im)
    x = int(0)
    y = int(0)
    w = int(w/2)
    h = int(h/2)
    return get_cropped_image(im, x,y,w,h,0)

def show_image(im, title):
    plt.figure(figsize=(10,5))    
    plt.imshow(im)
    plt.title(title)
    plt.show()

def get_yes_checkbox(c):
    return c[0]

def get_no_checkbox(c):
    return c[1]

def get_countours_number_inside(c):
    return c[1]

def get_darkness_level(c):
    return c[2]

def get_yes2no_darkness_ratio(c):
    ratio = 10000000
    yes_darkness = get_darkness_level(get_yes_checkbox(c))
    no_darkness = get_darkness_level(get_no_checkbox(c))
    if no_darkness != 0:
        ratio = yes_darkness/no_darkness
    return ratio      

def get_no2yes_darkness_ratio(c):
    ratio = 10000000
    yes_darkness = get_darkness_level(get_yes_checkbox(c))
    no_darkness = get_darkness_level(get_no_checkbox(c))
    if yes_darkness != 0:
        ratio = no_darkness/yes_darkness
    return ratio                                                                           
                                                                          
def check_if_yes_result(c):                                                                                   
    if (get_countours_number_inside(get_yes_checkbox(c)) > get_countours_number_inside(get_no_checkbox(c))) or ((get_yes2no_darkness_ratio(c) > 3) and (get_darkness_level(get_yes_checkbox(c)) > 0.3)):
            return True
    return False

def check_if_no_result(c):
    if (get_countours_number_inside(get_no_checkbox(c)) > get_countours_number_inside(get_yes_checkbox(c))) or ((get_no2yes_darkness_ratio(c) > 3) and (get_darkness_level(get_no_checkbox(c)) > 0.3)):
        return True
    return False

def check_if_none_result(c):
    if (get_countours_number_inside(get_yes_checkbox(c)) == 0) and (get_countours_number_inside(get_no_checkbox(c)) == 0) and (get_darkness_level(get_yes_checkbox(c)) < 0.3) and (get_darkness_level(get_no_checkbox(c)) < 0.3):
        return True
    return False

def check_if_both_result(c):
    if (get_countours_number_inside(get_yes_checkbox(c)) != 0) and (get_countours_number_inside(get_no_checkbox(c)) != 0) and (get_darkness_level(get_yes_checkbox(c)) >= 0.3) and (get_darkness_level(get_no_checkbox(c)) >= 0.3):
        return True
    return False

def get_vote(cb):
    # cb [Yes/No][Y position][Number Of Contours Inside][Dark level]
    vote = 'undefined'
    cb.sort()
    
    if len(cb) == 2:
        # check by number of countours and darkness, where [0,n] is Yes, [1,n] is No
        if check_if_yes_result(cb):
            return 'yes'
        else: 
            if check_if_no_result(cb):
                return 'no'
            else:
                if check_if_none_result(cb):
                    return 'none'
                else:
                    if check_if_both_result(cb):
                        return 'both'               
    return vote

def get_answers(image,crop,cropRight,cropBottom):
    
    im_cropped = crop_image(image,crop,cropRight,cropBottom)

    im_normalized = normalize(im_cropped)

    contours = get_contours(im_normalized)
    
    checkbox_contours = []
    checkbox_insights = []

    # Loop over each and every contours for filtering the shape
    
    for c in contours:
        if isSquare(c):    
            # Get cropped checkbox and analyse if it marked
            x,y,w,h = get_square_info(c)
            
            checkbox_crop = get_cropped_image(im_normalized, x,y,w,h,5)
            contours_inside, dark_level = get_square_insights(checkbox_crop)   
            
            checkbox_insights.append([y,contours_inside,dark_level])
            
            # Draw boundaries on shapes found
            cv2.rectangle(im_cropped, (x, y), (x + w, y + h), (36,255,12), 2)
            checkbox_contours.append(c)
    
    return checkbox_insights

def try_to_get_answers_from_right_bottom_image_part(im_orig):
    im = im_orig.copy()
    return get_answers(im,True,True,True)

def try_to_get_answers_from_right_upper_image_part(im_orig):
    im = im_orig.copy()
    return get_answers(im,True,True,False)

def try_to_get_answers_from_left_bottom_image_part(im_orig):
    im = im_orig.copy()
    return get_answers(im,True,False,True)

def try_to_get_answers_from_left_upper_image_part(im_orig):
    im = im_orig.copy()
    return get_answers(im,True,False,False)

def rotate_document(im_orig):
    img_rotated = im_orig.copy()
    checkbox_insights = try_to_get_answers_from_right_bottom_image_part(im_orig)
    if len(checkbox_insights) == 0:
        checkbox_insights = try_to_get_answers_from_right_upper_image_part(im_orig)
        if len(checkbox_insights) != 0:
            img_rotated = rotate_image_90(im_orig)
        else:
            checkbox_insights = try_to_get_answers_from_left_upper_image_part(im_orig)
            if len(checkbox_insights) != 0:
                img_rotated = rotate_image_180(im_orig) 
            else:
                checkbox_insights = try_to_get_answers_from_left_bottom_image_part(im_orig)
                if len(checkbox_insights) != 0:
                    img_rotated = rotate_image_270(im_orig)
    return img_rotated

def rotate_image_90(img):
    img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img_rotated

def rotate_image_180(img):
    img_rotated = cv2.rotate(img, cv2.ROTATE_180)
    return img_rotated

def rotate_image_270(img):
    img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_rotated

def get_signature(image_orig,crop,cropRight,cropBottom):
    
    im_cropped = crop_image(image_orig,crop,cropRight,cropBottom)
    im_cropped = crop_image(im_cropped,crop,cropRight,cropBottom)
    
    img = cv2.threshold(im_cropped, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

    # connected component analysis by scikit-learn framework
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    the_biggest_component = 0
    total_area = 0
    counter = 0
    average = 0.0
    for region in regionprops(blobs_labels):
        if (region.area > 10):
            total_area = total_area + region.area
            counter = counter + 1
        # take regions with large enough areas
        if (region.area >= 250):
            if (region.area > the_biggest_component):
                the_biggest_component = region.area

    average = (total_area/counter)

    # experimental-based ratio calculation, modify it for your cases
    # a4_constant is used as a threshold value to remove connected pixels
    # are smaller than a4_constant for A4 size scanned documents
    a4_constant = ((average/84.0)*250.0)+100

    # remove the connected pixels are smaller than a4_constant
    b = morphology.remove_small_objects(blobs_labels, a4_constant)
    # save the the pre-version which is the image is labelled with colors
    # as considering connected components

    # save and read the pre-version
    filename = str(uuid.uuid4()) + '.png'
    signedFileUrl = upload_to_file_storage(b,'signs',filename)

    #cv2.imwrite(filename, b)
    #img = cv2.imread(filename, 0)

    #if os.path.exists(filename):
    #    os.remove(filename)
    #img = cv2.imread(signedFileUrl,0)
    img = url_to_image(signedFileUrl)
    # ensure binary
    delete_file_from_storage('signs',filename)

    img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # save the the result
    #show_image(img_thresh, 'output')
    #filename = str(uuid.uuid4()) + '.png'
    #signedFileUrl = upload_to_file_storage(img_thresh,'signs',filename)

    contours = get_contours(img_thresh)
    isSigned = len(contours) > 0
    
    return isSigned

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
	# return the image
	return image

def get_if_document_valid(vote,isSigned):

    if isSigned == False:
        return False
    else:
        if (vote == 'yes') or (vote == 'no'):
            return True
    return False

def get_bytes_from_image(data):
    is_success, im_buf_arr = cv2.imencode(".jpg", data)
    byte_im = im_buf_arr.tobytes()
    return byte_im

def upload_files_to_storage(im_orig,im_rotated,isValid,vote):
    name_id = str(uuid.uuid4())
    filename =  name_id + '.jpg'
    origFileUrl = upload_to_file_storage(im_orig,'originals',filename)
    voteRecognizedFileUrl = upload_to_file_storage(im_rotated,'valid-'+str(isValid).lower()+'-'+'vote-'+vote,filename)
    return origFileUrl,voteRecognizedFileUrl,name_id
    
def upload_to_file_storage(data,container_name,dest_file_name):
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=dest_file_name)
    blob_client.upload_blob(get_bytes_from_image(data))
    return blob_client.url

def delete_file_from_storage(container_name,file_name):
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
    blob_client.delete_blob()

def save_vote_to_azure_table(vote, isSigned, isValid, origFileUrl,voteRecognizedFileUrl,lat,lng,date_time_original,name_id):
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    table_service = TableService(connection_string=connect_str)
    task = create_task(vote, isSigned, isValid, origFileUrl,voteRecognizedFileUrl,lat,lng,date_time_original,name_id)
    table_service.insert_entity('voteresults', task)

def create_task(vote, isSigned, isValid, origFileUrl,voteRecognizedFileUrl,lat,lng,date_time_original,name_id):
    task = Entity()
    task.PartitionKey = 'id'
    task.RowKey = name_id
    task.vote = vote
    task.signed = isSigned
    task.valid = isValid
    task.origFileUrl = origFileUrl
    task.voteRecognizedFileUrl = voteRecognizedFileUrl
    task.latitude = lat
    task.longitude = lng
    task.dateTimeOriginal = date_time_original
    return task


def get_vote_from_document(img_bytes):
    
    lat,lng,date_time_original = get_meta_data_from_image(img_bytes)

    im_orig = convert_img_bytes_to_cv2_img(img_bytes)
    
    #im_orig = cv2.imread(source_file)
    
    #show_image(im_orig, 'original image: ' + source_file)
    
    img_warpped = warp_image(im_orig)
    
    im_rotated = rotate_document(img_warpped)
    
    checkbox_insights = get_answers(im_rotated,True,True,True)
    
    #show_image(im_rotated, 'checkboxes_image: ' + source_file)
    
    vote = get_vote(checkbox_insights)
    
    #print(vote)
    
    isSigned = get_signature(im_rotated,True,True,False)

    isValid = get_if_document_valid(vote,isSigned)

    origFileUrl,voteRecognizedFileUrl,name_id = upload_files_to_storage(im_orig,im_rotated,isValid,vote)

    save_vote_to_azure_table(vote, isSigned, isValid, origFileUrl,voteRecognizedFileUrl,lat,lng,date_time_original,name_id)

    message = prepare_json_message(vote,isSigned, isValid, origFileUrl,voteRecognizedFileUrl,lat,lng,date_time_original)

    return message
        
        