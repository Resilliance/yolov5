from threading import local
import torch
from PIL import Image 
from helpers import draw_box, url_to_img, img_to_bytes


class PythonPredictor:

    def __init__(self, config):
        """ Download pretrained model. """
        #self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()
        self.model = torch.hub.load('Resilliance/yolov5', 'best19.19.11',pretrained=True).autoshape()  # default
        #self.model.load_state_dict(torch.load('best19.19.11.pt'))

    def predict(self, payload):
        """ Run a model based on url input. """

        # Inference
        self.model.conf = 0.5 # confidence 0-1
        img = url_to_img(payload["url"])
        results = self.model(img)

        # Draw boxes
        boxes = results.xyxy[0].numpy()
        box_img = draw_box(img, boxes)

        # Save image
        #box_img.save("sample_data/output.png", "PNG")

        return img_to_bytes(box_img)
