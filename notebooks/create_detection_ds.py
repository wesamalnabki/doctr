from datasets import load_dataset 
from pdf2image import convert_from_bytes
from PIL import ImageDraw
import uuid 
import os 
import json 
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

VAL_SIZE = 0.02
MAX_PAGES_COUNT =  100_000 # None to ignore 

detection_root_images = r"/opt/walnabki/OCR_Dataset/detection_ds/images"
os.makedirs(detection_root_images, exist_ok=True)
detection_root_labels = r"/opt/walnabki/OCR_Dataset/detection_ds/labels"
os.makedirs(detection_root_labels, exist_ok=True)


detct_train_root_images = r"/opt/walnabki/OCR_Dataset/detection_train/images"
os.makedirs(detct_train_root_images, exist_ok=True)
detct_train_root_labels = r"/opt/walnabki/OCR_Dataset/detection_train/labels"
os.makedirs(detct_train_root_labels, exist_ok=True)

train_json_path = r'/opt/walnabki/OCR_Dataset/detection_train/labels.json'

detct_val_root_images = r"/opt/walnabki/OCR_Dataset/detection_val/images"
os.makedirs(detct_val_root_images, exist_ok=True)
detct_val_root_labels = r"/opt/walnabki/OCR_Dataset/detection_val/labels"
os.makedirs(detct_val_root_labels, exist_ok=True)

val_json_path = r'/opt/walnabki/OCR_Dataset/detection_val/labels.json'

dataset = load_dataset('pixparse/pdfa-eng-wds', streaming=True )


def adjust_cord(bbox, img_w, img_h):
    x, y, w, h = bbox
    x1, y1 = x, y
    x2, y2 = x + w, y
    x3, y3 = x + w, y + h
    x4, y4 = x, y + h

    return [
        [int(x1 * img_w), int(y1 * img_h)],
        [int(x2 * img_w), int(y2 * img_h)],
        [int(x3 * img_w), int(y3 * img_h)],
        [int(x4 * img_w), int(y4 * img_h)]
    ]

def build_page_input(image_file, bboxs):
    fn = str(uuid.uuid4())
    img_name = fn+".png"
    json_name = fn+".json"
    image_file.save(os.path.join(detection_root_images, img_name))
    bboxs = [adjust_cord(bbox, image_file.size[0], image_file.size[1]) for bbox in bboxs]
    data = {
        fn +".png": {
        'img_dimensions': image_file.size,
        'img_hash': fn,
        'polygons': bboxs
     }
    }
    with open(os.path.join(detection_root_labels, json_name), 'w') as outfile:
        json.dump(data, outfile, indent=2)


if __name__=="__main__":

    print("Start creating detection dataset")
    ctr = 0

    for sample in iter(dataset['train']):
        
        if MAX_PAGES_COUNT and ctr > MAX_PAGES_COUNT:
            break

        pdf = sample['pdf']
        pages  =sample['json']['pages']
        pages_imgs = convert_from_bytes(pdf)

        for page_id in range(len(pages_imgs)): 

            ctr +=1
            print("Process pages: ", ctr)         
            try:
                words = pages[page_id]['words']['text']
                bboxs = pages[page_id]['words']['bbox']
                
                if len(words)!=len(bboxs):
                    print("MASSIVE ERROR --> ignore page")
                    continue

                images_bbox  = pages[page_id]['words']['bbox']
                page_img = pages_imgs[page_id]

                if bboxs:
                    build_page_input(image_file = page_img, bboxs = bboxs)
                else:
                    print("no bboxs--> escape")
                    continue        
            except Exception as ex:
                print("some exception, escape")
    
    print("Split the dataset")
    img_names = [x[:-4] for x in os.listdir(detection_root_images)]
    x_train ,x_test = train_test_split(img_names,test_size=VAL_SIZE, random_state=42) 
    len(x_train), len(x_test)

    print("Moving imgs to val folder")
    for fn_test in tqdm(x_test):
        src_lab = os.path.join(detection_root_labels, fn_test + ".json") 
        des_lab = os.path.join(detct_val_root_labels, fn_test + ".json") 

        src_img = os.path.join(detection_root_images, fn_test + ".png") 
        des_img = os.path.join(detct_val_root_images, fn_test + ".png")

        shutil.move(src_lab, des_lab) 
        shutil.move(src_img, des_img)

    print("Creating labels file for training")
    all_data_val = dict()
    for f in os.listdir(detct_val_root_labels):
        with open(os.path.join(detct_val_root_labels , f)) as json_file:
            data = json.load(json_file)
            all_data_val.update(data) 


    with open(val_json_path, 'w') as outfile:
        json.dump(all_data_val, outfile, indent=2)

    
    print("Moving imgs to training folder")
    for fn_train in tqdm(x_train):
        src_lab = os.path.join(detection_root_labels, fn_train + ".json") 
        des_lab = os.path.join(detct_train_root_labels, fn_train + ".json") 

        src_img = os.path.join(detection_root_images, fn_train + ".png") 
        des_img = os.path.join(detct_train_root_images, fn_train + ".png")

        shutil.move(src_lab, des_lab) 
        shutil.move(src_img, des_img)

    print("Creating labels file for training")
    all_data_train = dict()
    for f in os.listdir(detct_train_root_labels):
        with open(os.path.join(detct_train_root_labels , f)) as json_file:
            data = json.load(json_file)
            all_data_train.update(data) 


    with open(train_json_path, 'w') as outfile:
        json.dump(all_data_train, outfile, indent=2)


