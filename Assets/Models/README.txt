Drop yolo11n-pose.onnx in this folder.

How to get the model:

  pip install ultralytics
  python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt').export(format='onnx', opset=15, imgsz=640, dynamic=False, simplify=True)"

This produces yolo11n-pose.onnx (~6 MB).

Unity will import it as a Sentis ModelAsset.
Drag it into the "Model Asset" slot on the YoloPoseDetector component.

Output shape expected: (1, 56, 8400)
  channels 0..3   = bbox cx, cy, w, h (pixels, 640x640 input)
  channel 4       = person confidence (0..1)
  channels 5..55  = 17 keypoints x (x, y, visibility)

If your export differs (e.g. transposed to [1, 8400, 56]), re-export with the
command above, or transpose in your model wrapper before passing to the detector.
