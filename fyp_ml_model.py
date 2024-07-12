import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import SamModel, SamProcessor
import matplotlib.pyplot as plt

class KinectHandler(object):
    def __init__(self):
        self.kinectd = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
        self.depth_width, self.depth_height = self.kinectd.depth_frame_desc.Width, self.kinectd.depth_frame_desc.Height
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        self.color_width, self.color_height = self.kinect.color_frame_desc.Width, self.kinect.color_frame_desc.Height

    def get_depth_frame(self):
        if self.kinectd.has_new_depth_frame():
            depth_frame = self.kinectd.get_last_depth_frame()
            depth_frame = depth_frame.reshape((self.depth_height, self.depth_width)).astype(np.uint16)
            return depth_frame
        return None

    def get_color_frame(self):
        if self.kinect.has_new_color_frame():
            color_frame = self.kinect.get_last_color_frame()
            color_frame = color_frame.reshape((self.color_height, self.color_width, 4))
            # Keep only RGB channels
            color_frame = color_frame[:, :, :3]
            return color_frame
        return None

    def close(self):
        self.kinect.close()

def main():

    # Load the processor and model
    print("owlvit model is loaded")
    owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    texts = [["water bottle"]]
    # Use GPU if available
    if torch.cuda.is_available():
        print("gpu is available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Set model in evaluation mode
    owl_model = owl_model.to(device)
    owl_model.eval()

    # Define a dictionary to map labels to colors https://matplotlib.org/stable/gallery/color/named_colors.html
    label_colors = {
        0: "red",  # cat
        1: "green",  # dog
        2: "blue",  # remote control
        3: "yellow",
        4: "darkorange",
        5: "brown",
        6: "cyan",
        7: "indigo",
        8: "lime"
    }

    kinect = KinectHandler()

    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

    def show_boxes_on_image(raw_image, boxes):
        plt.figure(figsize=(10,10))
        plt.imshow(raw_image)
        for box in boxes:
            show_box(box, plt.gca())
        plt.axis('on')
        plt.show()

    def show_points_on_image(raw_image, input_points, input_labels=None):
        plt.figure(figsize=(10,10))
        plt.imshow(raw_image)
        input_points = np.array(input_points)
        if input_labels is None:
            labels = np.ones_like(input_points[:, 0])
        else:
            labels = np.array(input_labels)
        show_points(input_points, labels, plt.gca())
        plt.axis('on')
        plt.show()

    def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
        plt.figure(figsize=(10,10))
        plt.imshow(raw_image)
        input_points = np.array(input_points)
        if input_labels is None:
            labels = np.ones_like(input_points[:, 0])
        else:
            labels = np.array(input_labels)
        show_points(input_points, labels, plt.gca())
        for box in boxes:
            show_box(box, plt.gca())
        plt.axis('on')
        plt.show()


    def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
        plt.figure(figsize=(10,10))
        plt.imshow(raw_image)
        input_points = np.array(input_points)
        if input_labels is None:
            labels = np.ones_like(input_points[:, 0])
        else:
            labels = np.array(input_labels)
        show_points(input_points, labels, plt.gca())
        for box in boxes:
            show_box(box, plt.gca())
        plt.axis('on')
        plt.show()


    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


    def show_masks_on_image(raw_image, masks, scores):
        if len(masks.shape) == 4:
            masks = masks.squeeze()
        if scores.shape[0] == 1:
            scores = scores.squeeze()

        nb_predictions = scores.shape[-1]
        fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask = mask.cpu().detach()
            axes[i].imshow(np.array(raw_image))
            show_mask(mask, axes[i])
            axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
            axes[i].axis("off")
        plt.show()
    print("sam model is loaded")
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    model.eval()

    while True:
        color_frame = kinect.get_color_frame()



        if color_frame is not None:

            # Process the image and text
            print("owl-processor")
            inputs = owl_processor(text=texts, images=color_frame, return_tensors="pt").to(device)
            outputs = owl_model(**inputs)

            # Get target sizes and process object detection results
            target_sizes = torch.Tensor([color_frame.shape[:2]])
            results = owl_processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.1
            )

            # Retrieve predictions for the first image
            i = 0
            text = texts[i]
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

            # Create a draw object
            color_frame_pil = Image.fromarray(color_frame)
            draw = ImageDraw.Draw(color_frame_pil)

            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                label_text = f"{text[label.item()]}: {round(score.item(), 2)}"
                print(f"Detected {label_text} at location {box}")

                # Convert the label tensor to an integer
                label_int = label.item()

                # Draw the bounding box with a specific color
                color = label_colors.get(label_int, "black")  # Use "black" as a default color if label_int is not in label_colors
                draw.rectangle(box, outline=color, width=2)

                # Calculate text size
                text_bbox = draw.textbbox((0, 0), label_text)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Draw a filled rectangle behind the text
                draw.rectangle(
                    [box[0], box[1] - text_height, box[0] + text_width, box[1]],
                    fill=color
                )
                draw.text((box[0], box[1] - text_height), label_text, fill="white")

            # Display the image using matplotlib
            plt.figure(figsize=(10, 10))
            plt.imshow(color_frame_pil)
            plt.axis("off")
            plt.show()
            # cv2.imshow(color_frame_pil)
            if box is not None:
                print("sam-processor started")
                inputs = processor(color_frame, return_tensors="pt").to(device)
                image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
                print("sam processor finished")

                input_boxes = [[box]]
                print("sam model is running")
                inputs = processor(color_frame_pil, input_boxes=[input_boxes], return_tensors="pt").to(device)

                inputs.pop("pixel_values", None)
                inputs.update({"image_embeddings": image_embeddings})

                with torch.no_grad():
                    outputs = model(**inputs)

                masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
                scores = outputs.iou_scores

                show_masks_on_image(color_frame_pil, masks[0], scores)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    kinect.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()