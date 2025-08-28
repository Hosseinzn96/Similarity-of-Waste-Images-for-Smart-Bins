# Similarity-of-Waste-Images-for-Smart-Bins
Deep learning framework for detecting object-level changes in smart-bin waste images using triplet learning and CNN backbones.

This repository contains the implementation of a deep learning framework for 
**object-level change detection in waste-bin images**. The system leverages 
polygon-based cropping, triplet loss learning, and multiple CNN backbones 
(ResNet-50/101, MobileNetV2, Xception) to detect and evaluate added or 
removed objects in smart-bin scenarios.

## Features
- Polygon-fit object cropping from COCO-style annotations
- Dynamic triplet generation with hard negative mining
- Training pipeline with TensorFlow/Keras
- Evaluation on Matching & Added-object detection tasks
- Supports multiple backbones: ResNet-50, ResNet-101, MobileNetV2, Xception
- Visualization tools for side-by-side inference
