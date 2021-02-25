# simple_fcos_pytorch
A simple implementation of Fully Convolutional One-Stage for Object Detection (FCOS) in pytorch, written in classic training loop style.

****
**More Details:**

Backbone: Resnet50

FPN: 5 outputs: p3, p4, p5, p6, p7 at stride 8,16,32,64,128 respectively

Head: Simple Convolution with stride 1

Data Format: Same as Scaled Yolo-v4, which is: c,x,y,w,h. where:
- c is class id,
-  x,y is the box center normalized coordinate,
-  w,h is the width and height of the box in normalized scale

*To do:*

- Add demo code to get bounding boxes results
- Refactor and add comment on the code
