import os
import pickle
import cv2 as cv

im_size = 5
path = 'data/analysis'


def mouse_event(event, x, y, flags, param):
    """"Click left button to remove image"""
    global val
    if event == cv.EVENT_LBUTTONDOWN:
        val = 1


img_paths = os.listdir(path)
out_list = []
for i in range(len(img_paths)):
    name = img_paths[i]
    current_path = f'{path}/{name}'
    img = cv.imread(f'{current_path}/circles.png')

    val = 0
    cv.imshow(name, img)
    cv.setMouseCallback(name, mouse_event)
    cv.waitKey(0)
    cv.destroyAllWindows()
    out_list.append(val)

print('Wrong images:')
for i in range(len(out_list)):
    if out_list[i] == 1:
        print(f'{img_paths[i]}')

pickle.dump(out_list, open(f'data/incorrect.p', "wb"))
