import numpy as np
import cv2
from PIL import ImageGrab as ig

while(True):
    screen = ig.grab(bbox=(30,100,1500,870))
    img_np = np.array(screen)
    RGB_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    cv2.imshow("test", np.array(RGB_img))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break