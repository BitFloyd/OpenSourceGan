'''
First save all images having our interested class in a folder for easy access
'''
import json
import os
import queue
import config
import utils
import threading
# Create a dictionary for converting the display name to labelname.
displayNameToLabelNameDict = utils.get_displayNameToLabelNameDict()
labelNameToDisplayNameDict = {value: key for key, value in displayNameToLabelNameDict.items()}

# TODO Make this a script argument.
interested_class = config.INTERESTED_CLASS

# Check if the intersted_class is in our dict. If not, suggest alternates and quit.
if (interested_class not in displayNameToLabelNameDict):
    utils.suggestAlternates(interested_class, displayNameToLabelNameDict)

# If interested_class is found, we should get the ids of all the classes and its specific subsets.
set_labelnames = set(utils.getClassAndSubsets(displayNameToLabelNameDict[interested_class]))

# For each of the labelnames, we need to find the images and the relevant bboxes.
imagesWithLabelsBboxes = utils.getImagesWithLabelsandBoxes(set_labelnames)
print("FOUND {dlen} IMAGES".format(dlen=len(imagesWithLabelsBboxes)))

#Get URLS for the relevant images.
imageIDtoUrlDict = utils.getUrlsForImages(imagesWithLabelsBboxes.keys())

save_path = os.path.join(config.IMAGES_ROOT, interested_class)
os.makedirs(save_path, exist_ok=True)

#Download the images into local.
out_queue = queue.Queue()

for imageID in imageIDtoUrlDict:
    uri = imageIDtoUrlDict[imageID]
    uri_to_filename = uri.replace('/', '_')
    uri_to_filename = uri_to_filename.replace(':', '_')
    list_bboxes = imagesWithLabelsBboxes.get(imageID,[])
    image_save_path = os.path.join(save_path, uri_to_filename)
    imagesWithLabelsBboxes[imageID] = {'filepath': image_save_path,'list_bboxes':list_bboxes}
    cmd = (uri, image_save_path)
    out_queue.put(cmd)

#Save this dict
with open(os.path.join(config.IMAGES_DICT_FOR_LABEL_META, interested_class + '_images_dict.json'), 'w') as f:
    json.dump(imagesWithLabelsBboxes, f)

threads = [threading.Thread(target=utils.download_func, args=(out_queue,)) for i in range(10)]
for thread in threads:
    thread.start()
