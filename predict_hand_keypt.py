from PIL import Image
from hand_keypoint import predict
import matplotlib.pyplot as plt

pre=predict.Predict()
im1 = Image.open(r"C:\Users\lahir\Downloads\hand1.jpg") 
im2 = Image.open(r"C:\Users\lahir\Downloads\hand2.jpg") 
im3 = Image.open(r"C:\Users\lahir\Downloads\hand3.jpg") 
im_list=[im1,im2,im3]
heatmap_np,joints_2d_right,joints_2d_left=pre.make_prediction(im_list)
n=2
im_2d=pre.draw_skeleton(im_list[n],joints_2d_right[n],joints_2d_left[n],['right'])
plt.imshow(im_2d)
plt.show()

plt.imshow(heatmap_np[n])
plt.show()