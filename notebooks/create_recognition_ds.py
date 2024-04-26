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
MAX_PAGES_COUNT =  5_000_000 # None to ignore 

recognition_root_images = r"/opt/walnabki/OCR_Dataset/recognition_ds/images"
os.makedirs(recognition_root_images, exist_ok=True)
recognition_root_labels = r"/opt/walnabki/OCR_Dataset/recognition_ds/labels"
os.makedirs(recognition_root_labels, exist_ok=True)


recognition_train_root_images = r"/opt/walnabki/OCR_Dataset/recognition_train/images"
os.makedirs(recognition_train_root_images, exist_ok=True)
recognition_train_root_labels = r"/opt/walnabki/OCR_Dataset/recognition_train/labels"
os.makedirs(recognition_train_root_labels, exist_ok=True)

train_json_path = r'/opt/walnabki/OCR_Dataset/recognition_train/labels.json'

recognition_val_root_images = r"/opt/walnabki/OCR_Dataset/recognition_val/images"
os.makedirs(recognition_val_root_images, exist_ok=True)
recognition_val_root_labels = r"/opt/walnabki/OCR_Dataset/recognition_val/labels"
os.makedirs(recognition_val_root_labels, exist_ok=True)

val_json_path = r'/opt/walnabki/OCR_Dataset/recognition_val/labels.json'

dataset = load_dataset('pixparse/pdfa-eng-wds', streaming=True )


def adjust_bbox(word_bbox, img):
    
    xr = word_bbox[0]*img.size[0]
    yr = word_bbox[1]*img.size[1] 
    w = word_bbox[2]*img.size[0]
    h = word_bbox[3]*img.size[1]
    xl = xr + w 
    yl = yr + h
    return [xr,yr, xl, yl]



if __name__=="__main__":

    # create ds 
    ctr = 0
    for sample in iter(dataset['train']):
            
        if MAX_PAGES_COUNT and ctr > MAX_PAGES_COUNT:
            break

        pdf = sample['pdf']
        pages  =sample['json']['pages']
        pages_imgs = convert_from_bytes(pdf)

        for page_id in range(len(pages_imgs)): 
            
            ctr +=1
            words = pages[page_id]['words']['text']
            bboxs = pages[page_id]['words']['bbox']
            page_img = pages_imgs[page_id]

            for word, bbox in zip(words, bboxs):
                fn = str(uuid.uuid4()) 
                page_labels_dict = dict()
                try:
                    bbox_adj = adjust_bbox(word_bbox=bbox, img = page_img)
                    cropped_img = page_img.crop(bbox_adj)
                    cropped_img.save(os.path.join(recognition_root_images, fn + ".png"))
                    page_labels_dict[fn+ ".png"] = word
                    with open(os.path.join(recognition_root_labels, fn + ".json"), 'w') as outfile:
                        json.dump(page_labels_dict, outfile, indent=2)
                except:
                    print(f"escape word --> {word}")
                    continue
        
    print("Split the dataset")
    img_names = [x[:-4] for x in os.listdir(recognition_root_images)]
    x_train ,x_test = train_test_split(img_names,test_size=VAL_SIZE, random_state=42) 
    len(x_train), len(x_test)


    print("Moving imgs to val folder")
    for fn_test in tqdm(x_test):
        src_lab = os.path.join(recognition_root_labels, fn_test + ".json") 
        des_lab = os.path.join(recognition_val_root_labels, fn_test + ".json") 

        src_img = os.path.join(recognition_root_images, fn_test + ".png") 
        des_img = os.path.join(recognition_val_root_images, fn_test + ".png")

        shutil.move(src_lab, des_lab) 
        shutil.move(src_img, des_img)

    print("Creating labels file for training")
    all_data_val = dict()
    for f in os.listdir(recognition_val_root_labels):
        with open(os.path.join(recognition_val_root_labels , f)) as json_file:
            data = json.load(json_file)
            all_data_val.update(data) 


    with open(val_json_path, 'w') as outfile:
        json.dump(all_data_val, outfile, indent=2)


    print("Moving imgs to training folder")
    for fn_train in tqdm(x_train):
        src_lab = os.path.join(recognition_root_labels, fn_train + ".json") 
        des_lab = os.path.join(recognition_train_root_labels, fn_train + ".json") 

        src_img = os.path.join(recognition_root_images, fn_train + ".png") 
        des_img = os.path.join(recognition_train_root_images, fn_train + ".png")

        shutil.move(src_lab, des_lab) 
        shutil.move(src_img, des_img)

    print("Creating labels file for training")
    all_data_train = dict()
    for f in os.listdir(recognition_train_root_labels):
        with open(os.path.join(recognition_train_root_labels , f)) as json_file:
            data = json.load(json_file)
            all_data_train.update(data) 


    with open(train_json_path, 'w') as outfile:
        json.dump(all_data_train, outfile, indent=2)



