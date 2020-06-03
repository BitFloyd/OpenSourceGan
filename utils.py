import csv
import json
import queue
import subprocess
import sys

import wget
from fuzzywuzzy import process
from urllib.error import HTTPError,  URLError
import config


def get_displayNameToLabelNameDict():
    displayNameToLabelNameDict = {}
    with open(config.IMAGE_CLASS_DESCRIPTIONS_FILE) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if (idx == 0):
                continue
            displayNameToLabelNameDict[row[1]] = row[0]
    return displayNameToLabelNameDict


def suggestAlternates(interested_class, displayNameToLabelNameDict):
    fuzzyratios = process.extract(interested_class, list(displayNameToLabelNameDict.keys()))
    print("THE CLASS YOU REQUESTED COULD NOT BE FOUND...HOWEVER, I FOUND THESE CLASSES THAT ARE SIMILAR.....")
    print([i[0] for i in fuzzyratios[0:5]])
    sys.exit(0)


def digDeep(list_to_add, labelname, level=0, inside=False, subcategory_list=None):
    if (subcategory_list == None):
        return

    labelnames_in_this_level = [i['LabelName'] for i in subcategory_list]

    if (not inside):
        # Search for the relevant name
        if (labelname in labelnames_in_this_level):
            # print("FOUND LABELNAME IN LEVEL : ", level)
            labelname_index = labelnames_in_this_level.index(labelname)
            list_to_add.append(labelname)
            digDeep(list_to_add, labelname, level + 1, True, subcategory_list[labelname_index].get('Subcategory', None))

        else:
            for subcategory in subcategory_list:
                digDeep(list_to_add, labelname, level + 1, False, subcategory.get('Subcategory', None))
    else:
        for subcategory in subcategory_list:
            list_to_add.append(subcategory['LabelName'])
            digDeep(list_to_add, labelname, level + 1, True, subcategory.get('Subcategory', None))

    return


def getClassAndSubsets(label_name):
    with open(config.LABEL_HIERARCHY_JSON, 'r') as f:
        class_hierarchy = json.load(f)

    list_classes = []
    digDeep(list_classes, label_name, 0, False, class_hierarchy.get('Subcategory', None))
    return list_classes


def getImagesWithLabelsandBoxes(label_name_set):
    imagesWithLabelsBboxes = {}

    with open(config.TRAIN_BBOX_ANNOTATIONS_FILE) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if (idx == 0):
                continue

            ImageID, Source, LabelName, Confidence, XMin, XMax, YMin, YMax = row[0:8]
            if (LabelName in label_name_set):
                bbox_list_for_image = imagesWithLabelsBboxes.get(ImageID, [])
                bbox_list_for_image.append(
                    {'LabelName': LabelName, 'XMin': XMin, 'XMax': XMax, 'YMin': YMin, 'YMax': YMax})
                imagesWithLabelsBboxes[ImageID] = bbox_list_for_image
    return imagesWithLabelsBboxes


def getUrlsForImages(imageIDList):
    url_dict = {key: None for key in imageIDList}

    for url_file in config.LIST_IMAGE_URL_FILES:
        with open(url_file) as f:
            csv_reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(csv_reader):
                if (idx == 0):
                    continue
                ImageID = row[0]
                OriginalURL = row[2]
                if (ImageID in url_dict):
                    url_dict[ImageID] = OriginalURL

    return url_dict


def run_subprocess_cmd(cmd):
    '''
    Run a subprocess shell command
    Args:
        - cmd: string shell cmd to run
    Returns:
        (o_str, rc): Tuple
        - o_str: Output string from shell
    Raises exception on Failure and exits
        - rc : Int, return code
    '''
    o_str = None
    rc = 0
    try:
        o = subprocess.check_output(cmd, shell=True)
        o_str = o.decode('utf-8').strip()
        print("Subprocess cmd %s returned %s" % (cmd, o_str))
    except subprocess.CalledProcessError as cmd_exception:
        print("Subprocess cmd %s failed with output %s "
              "and Return code %d: " %
              (cmd, cmd_exception.output, cmd_exception.returncode))
        o_str = cmd_exception.output.decode('utf-8').strip()
        rc = cmd_exception.returncode
    return (o_str, rc)


def download_func(in_queue):
    while 1:
        try:
            cmd = in_queue.get(block=False)
        except queue.Empty:
            return
        else:
            try:
                wget.download(cmd[0], cmd[1])
            except HTTPError:
                print("Could not download ", cmd[0])
                continue
            except URLError:
                in_queue.put(cmd)
                #Will be processed later.
                continue


