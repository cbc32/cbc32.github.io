from rembg import remove

import cv2, os

folder_name = ['RESIZED_DATASET', 'RESIZED_TESTING_DATA']

folder_size = 38

for i in folder_name:

    for j in range(folder_size):

        iPath = i + '/' + str(j) + '/'

        oPath = i + '_PROCESSED' + '/' + str(j) + '/'

        isExist = os.path.exists(oPath)

        if not isExist:

            os.makedirs(oPath)

        for f in os.listdir(iPath):

            img = cv2.imread(os.path.join(iPath, f))

            img = remove(img)

            cv2.imwrite(os.path.join(oPath, f), img)