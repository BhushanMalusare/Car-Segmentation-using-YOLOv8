from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('runs/segment/train/weights/last.pt')

# Run inference on 'bus.jpg'
results = model('download.jpg')  # results list

# Show the results
for r in results:
    im_array = r.plot(boxes=False)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image

