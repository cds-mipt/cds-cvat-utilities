########################################
#    Please change the dataset         #
#   path(in main() function) and       #
#   json file name in 11th string      #
#   before using the program           #
########################################

import xml.etree.ElementTree as ET
import json
import os

labels = {} #global
with open("99_Taganrog-Day-Night_cropped_traffic_lights.json", "r") as file:
    data = json.load(file)

def create_header(im_num):
    root = ET.Element("annotations")
    prom = ET.Element("version")

    root.append(prom)
    prom.text = "1.0"

    meta = ET.Element("meta")
    root.append(meta)
    task = ET.Element("task")
    meta.append(task)

    prom = ET.Element("id")
    task.append(prom)

    prom = ET.Element("name")
    task.append(prom)

    prom = ET.Element("size")
    task.append(prom)

    ET.SubElement(task,"name")

    name = ET.Element( "name")

    ET.Element( "frame_filter")


    return root

def create_labels():
    global labels
    labelfile = open("labels.txt", "w")
    for elem in data["categories"]:
        labels[elem["id"]] = elem["name"]
        print('"'+elem["name"]+'"', file=labelfile)


    return labels

def points_layout(img, hash):

    #find annotation
    for annot in data["annotations"]:
        cnt = 0
        if (annot["image_id"] == hash and annot["iscrowd"] == 0):

            box_array = annot["bbox"]
            points = ""
            seg_len = len(annot["segmentation"][0])
            while(cnt < seg_len):
                points += str(annot["segmentation"][0][cnt])
                points += ","
                points += str(annot["segmentation"][0][cnt + 1])
                points += ";"
                cnt += 2
            label_id = annot["category_id"]
            conf = str(annot['conf'])

            polygon_sub_element = ET.SubElement(img, "polygon", {"label": labels[label_id], "occluded":"0",\
                                  "points":points[:-1], 'conf':conf})
            attribute_polygon_sub_element = ET.SubElement(polygon_sub_element, 'attribute', {})
            attribute_polygon_sub_element.text = str(conf)
            attribute_polygon_sub_element.set('name','conf')
            if (len(box_array) != 0):
                xtl = box_array[0]
                ytl = box_array[1]
                lenght = box_array[2]
                height = box_array[3]

                xbr = round(xtl+lenght, 2)
                ybr = round(ytl+height, 2)
                box_sub_element = ET.SubElement(img, "box", {"label": labels[label_id], "occluded":"0",\
                               "xtl":str(xtl), "ytl":str(ytl),\
                               "xbr":str(xbr), "ybr":str(ybr)})
                attribute_box_sub_element = ET.SubElement(box_sub_element, 'attribute', {})
                attribute_box_sub_element.text = str(conf)
                attribute_box_sub_element.set('name','conf')



def add_image(IMAGE, root, wdt, hght, f_name, id, tmp):
    img = ET.SubElement(root, "image", {"id":str(tmp),"name":f_name, "width":str(wdt),"height":str(hght)})

    points_layout(img, id)

def f_name(image):
    return image["file_name"]

def main():
    xml_f = open("99_Taganrog-Day-Night_cropped_traffic_lights.xml", "wb")

    im_number = len(data["images"])
    create_labels()
    root = create_header(im_number)

    file_list = list()
    directory = "/home/solomentsev_yaroslav/Jupyter_Notebook/datasets/Cropped_traffic_lights_TAGANROG/images"
    for filename in os.listdir(directory):
        file_list.append(filename)

    data["images"].sort(key=f_name)

    tmp = 0
    for image in data["images"]:
        if (image["file_name"] in file_list):
            add_image(image, root, image["width"], image["height"], image["file_name"], image["id"], tmp)
            create_labels()
            tmp += 1

    tree = ET.ElementTree(root)
    tree.write(xml_f)
    
    print(root[2].attrib)

if __name__ == "__main__":
    main()
