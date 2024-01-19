import os
import pandas as pd

DEBUG = True


def img_3_dir(img_dir):
    if img_dir[-1] == "/":
        img_dir.pop()
    return img_dir+'/nobody', img_dir+'/no_face', img_dir+'/people'


def reformat_img_name(img_dir):
    current = os.getcwd()
    for folder in img_3_dir(img_dir):
        os.chdir(current)
        mark = folder.split('/')[-1]
        os.chdir(folder)
        for img in os.listdir():
            if img[:6] != mark:
                os.rename(img, mark+img)
    os.chdir(current)


def gen_csv(img_dir, path_csv):
    current = os.getcwd()
    reformat_img_name(img_dir)
    list_dir = list(img_3_dir(img_dir))
    img_nobody = os.listdir(list_dir[0])
    img_no_face = os.listdir(list_dir[1])
    img_people = os.listdir(list_dir[2])
    list_labels = [0 for k in range(len(img_nobody))] + [1 for k in range(len(img_no_face))] + [2 for k in range(len(img_people))]
    img_list_name = img_nobody + img_no_face + img_people
    dict_data = {"name": img_list_name, "label": list_labels}

    csv = pd.DataFrame(data=dict_data)
    if DEBUG:
        print(csv)

    os.chdir(path_csv)
    csv.to_csv("labels.csv", index=False, header=False)

    os.chdir(current)
