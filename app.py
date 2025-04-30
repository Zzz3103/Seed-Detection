import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from yolov3.model import YOLOv3
from yolov3.utils import cells_to_bboxes, non_max_suppression
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.title("📸 Nhận diện hạt nảy mầm bằng YOLO")
uploaded_file = st.file_uploader("Chọn một ảnh để dự đoán", type=["jpg", "jpeg", "png"])
# col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # with col1:
    st.image(image, caption="Ảnh đã chọn", use_container_width=True)
    
    model_choice = st.radio(
        "Chọn model để dự đoán",
        ("🌱 Fine-tuned YOLOv11", "🛠️ From Scratch YOLOv3")
    )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.JPG') as tmp:
        image.save(tmp.name)
        
        
        
        yolo11_model = YOLO("models/yolov11.pt")
        results = yolo11_model(tmp.name)
        res_plotted = results[0].plot(conf=False, labels=False)
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        class_counts = Counter(cls_ids)
        total_seeds = sum(class_counts.values())
        germinated_seeds = class_counts.get(0, 0)
        germinated_rate = germinated_seeds / total_seeds * 100
        cv2.putText(
            res_plotted, 
            f"Germinated seeds: {germinated_seeds}/{total_seeds}",
            (100,200),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 5,  # scale
            color = (255, 0, 255),  # màu (BGR) - xanh lá
            thickness = 5,  # độ dày
            lineType = cv2.LINE_AA
        )
        
        cv2.putText(
            res_plotted, 
            f"Germination rate: {germinated_rate:.2f}%",
            (100,400),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 5,  # scale
            color = (255, 0, 255),  # màu (BGR) - xanh lá
            thickness = 5,  # độ dày
            lineType = cv2.LINE_AA
        )
        # with col2:
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        IMAGE_SIZE = 416
        S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
        ANCHORS = [
            [(0.0803, 0.0729), (0.0868, 0.0928), (0.0750, 0.0726)],
            [(0.0350, 0.0671), (0.0374, 0.0485), (0.0277, 0.0558)],
            [(0.0309, 0.0272), (0.0292, 0.029), (0.0221, 0.0151)],
        ]
        scaled_anchors = (
            torch.tensor(ANCHORS)
            * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(DEVICE)
        SEEDS_CLASSES = [
            ".",
            ","
        ]
        CHECKPOINT_FILE = "models/yolov3.pth.tar"
        
        def plot_image(image, boxes):
            """Plots predicted bounding boxes on the image"""
            cmap = plt.get_cmap("tab20b")
            class_labels = SEEDS_CLASSES
            colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
            im = np.array(image)
            height, width, _ = im.shape

            # Create figure and axes
            fig, ax = plt.subplots(1)
            # Display the image
            ax.imshow(im)

            # box[0] is x midpoint, box[2] is width
            # box[1] is y midpoint, box[3] is height

            # Create a Rectangle patch
            germinated_cnt = 0
            total_seeds = len(boxes)
            for box in boxes:
                assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
                class_pred = box[0]
                if class_pred == 0:
                    germinated_cnt += 1
                box = box[2:]
                upper_left_x = box[0] - box[2] / 2
                upper_left_y = box[1] - box[3] / 2
                rect = patches.Rectangle(
                    (upper_left_x * width, upper_left_y * height),
                    box[2] * width,
                    box[3] * height,
                    linewidth=1,
                    edgecolor=colors[int(class_pred)],
                    facecolor="none",
                )
                # Add the patch to the Axes
                ax.add_patch(rect)
            germinated_rate = germinated_cnt / total_seeds * 100
            ax.text(
                10,
                100,
                s=f"Germinated seeds: {germinated_cnt}/{total_seeds}",
                color="purple",
                fontdict={
                    'fontsize': 6,
                    'fontweight': 'bold',
                    'style': 'italic'
                }
            )
            ax.text(
                10,
                120,
                f"Germination rate: {germinated_rate:.2f}%",
                color='purple',
                fontdict={
                    'fontsize': 6,
                    'fontweight': 'bold',
                    'style': 'italic'
                }
            )
            st.pyplot(fig)
        
        model = YOLOv3(num_classes=2).to(DEVICE)
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        test_transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=IMAGE_SIZE),
                A.PadIfNeeded(
                    min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value = 0
                ),
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
                ToTensorV2(),
            ],
        )
        def plot_one_image(model, image_path, thresh, iou_thresh, anchors):
            model.eval()
            image = np.array((Image.open(image_path)).convert("RGB"))
            augmentations = test_transforms(image = image)
            image = augmentations["image"].unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                out = model(image)
                bboxes = [[] for _ in range(image.shape[0])]
                for i in range(3):
                    _, _, S, _, _ = out[i].shape
                    anchor = anchors[i]
                    boxes_scale_i = cells_to_bboxes(
                        out[i], anchor, S=S, is_preds=True
                    )
                    for idx, (box) in enumerate(boxes_scale_i):
                        bboxes[idx] += box

                model.train()

            nms_boxes = non_max_suppression(
                bboxes[0], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
            )
            plot_image(image[0].permute(1,2,0).detach().cpu(), nms_boxes)
            
        if model_choice == "🌱 Fine-tuned YOLOv11":
            st.image(res_plotted, caption="Kết quả dự đoán", use_container_width=True)
        else:
            plot_one_image(model, tmp.name, 0.65, 0.25, scaled_anchors)