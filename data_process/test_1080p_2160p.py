import numpy as np
img_1080p = np.zeros((1080, 1920))
scale = 3
img_ = np.zeros((1080*scale, 1920*scale))

new_K = K.copy()
new_K[:2] = new_K[:2] * scale
x1, y1, x2, y2 = get_3d_vertex(center, ref_lwh, yaw, trans_rfu2cam, new_K)
img_[y1:y2, x1:x2] = 255


center[1] = 20
x1, y1, x2, y2 = get_3d_vertex(center, ref_lwh, yaw, trans_rfu2cam, K)
scene_mask = np.zeros((IMG_H, IMG_W, 3), np.uint8)
scene_mask[y1:y2, x1:x2, :] = 255
tar_mask = scene_mask[:,:,0] > 128
print(tar_mask.sum())
tar_mask = tar_mask.astype(np.uint8)
gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)
cv2.imwrite("test__.png", gen_image[:, :, ::-1])


scale = 3
back_image = cv2.resize(back_image, (1920 * scale, 1080 * scale))

new_K = K.copy()
new_K[:2] = new_K[:2] * scale

center[1] = 20
x1, y1, x2, y2 = get_3d_vertex(center, ref_lwh, yaw, trans_rfu2cam, new_K)
scene_mask = np.zeros((IMG_H * scale, IMG_W * scale, 3), np.uint8)
scene_mask[y1:y2, x1:x2, :] = 255
tar_mask = scene_mask[:,:,0] > 128
print(tar_mask.sum())
tar_mask = tar_mask.astype(np.uint8)
gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)
cv2.imwrite("test__.png", gen_image[:, :, ::-1])