import cv2
import numpy as np
from datasets.private_data.controlnet_sd_private_dataset import ControlPrivate, PromptGenerator
from datasets.private_data.private.private_multimodal import PrivateMultiModalData


class ControlPrivateStatic(ControlPrivate):
    def __init__(self, *args, **kwargs):
        super(PrivateMultiModalData, self).__init__(*args, **kwargs)
        self.prompt_generator = PromptGenerator(empty_rate=0) # 模型自带drop
        self.img_key_list = self.image.camera_names
        self.class_names = self.annotation.class_names

    colors = np.array([
                [255, 255, 255],
                [128, 64, 128], # cone
                [244, 35, 232], # crash_bar
                [70, 70, 70], # water_horse
                [102, 102, 156], # bucket
                [190, 153, 153], # construction_sign
                [0, 165, 255], # triangle
            ])


    def draw_bboxes(self, target, bboxes, labels, json_path):
        img = np.zeros((target.shape[0], target.shape[1], 3))
        img = img.copy().astype(np.uint8)
        if labels is None or len(labels) == 0:
            return img

        for i in range(len(bboxes)):
            label = labels[i]
            label_name = self.class_names[label]
            xmin, ymin, xmax, ymax = bboxes[i]

            img[int(ymin) : int(ymax), int(xmin) : int(xmax),...] = self.colors[labels[i]+1]
            # cal area of 2d box
            lenth = ymax - ymin
            width = xmax - xmin
            area = (xmax - xmin) * (ymax - ymin)
            img_name = json_path.replace("/", "-")
            if lenth > 180 or width > 180:
                _x = (xmin + xmax) / 2.0
                _y = (ymin + ymax) / 2.0
                x = int(_x)
                y = int(_y)
                target_mask = np.zeros_like(target)
                target_mask[int(ymin) : int(ymax), int(xmin) : int(xmax),...] = 1
                thres = int(max(lenth, width) / 2.0)
                target_mask = target_mask[y-thres:y+thres, x-thres:x+thres, :]
                crop_img = target[y-thres:y+thres, x-thres:x+thres, :]
                import IPython; IPython.embed()
                # crop_img = target[int(ymin) : int(ymax), int(xmin) : int(xmax),...]
                crop_img *= target_mask
                try:
                    cv2.imwrite(f"./crop_obj_new/{img_name}_{label_name}_{int(area)}.png", crop_img)
                except:
                    pass
        return img 

    def _prepare_one_frame(self, item):

        if len(item['gt_labels']) == 0:
            target = item['imgs']
            source = np.zeros_like(target)
            return target, source
        source_img = item['imgs'] #torch.Size([6, 3, 256, 704])
        img_list = []
        static_list = []
        # 目前静态障碍物只标单v2d，这里模拟多v的shape
        source_label = item['gt_labels'][None,...] # torch.Size([1, n])  
        source_bbox = item['gt_boxes'][None,...] # torch.Size[1, n, 4]
        # 框的坐标反归一化
        source_bbox[...,[0,2]] = source_bbox[...,[0,2]] * source_img.shape[2]
        source_bbox[...,[1,3]] = source_bbox[...,[1,3]] * source_img.shape[1]
        for view_id, view in enumerate(self.img_key_list):
            img = source_img[view_id]

            bboxes2d = source_bbox[view_id]
            labels2d = source_label[view_id]

            static = self.draw_bboxes(img, bboxes2d, labels2d, item["json_path"])
            img_list.append(img)
            static_list.append(static)

        target = np.stack(img_list, 0)
        static = np.stack(static_list, 0)

        return target, static


if __name__ == "__main__":
    import random
    from tqdm import tqdm
    import cv2
    from datasets.private_data.data_cfg.E171_image_static_1v import static_dataset_cfg
    dataset = ControlPrivateStatic(**static_dataset_cfg)
    idxs = random.choices(list(range(len(dataset))), k=len(dataset))
    nums = 0
    for idx in tqdm(idxs, desc="search nonzero input"):
        data = dataset[idx]
        if (data["hint"] != -1).any():
            img = data["video"][0].permute(1,2,0).numpy()
            hint = data["hint"][0].permute(1,2,0).numpy()
            ret = np.concatenate((img, hint), axis=0)
            ret = (ret+1)*127.5
            cv2.imwrite("/gpfs/public-shared/fileset-groups/wheeljack/wentiancheng/ControlNeXt/ControlNet-SDXL/debug/%s.jpg"%nums,
                        ret.astype(np.uint8))
            nums+=1
    from IPython import embed; embed()