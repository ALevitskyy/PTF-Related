# PTF-Related

Repository for some of the code I did for Punch-to-Face project, which aims to build 3D reconstructions of MMA fights in Virtual Reality creating a new niche in entertainment industry.

The repository features:
1) **textures** folder contains code to convert DensePose outputs to Blender compatible textures for SMPL models
2) **canvas_segmentation** folder contains binary segmentation pipeline made using Catalyst
3) **canvas_segmentation/data/augmentations/** contains some customary augmentations (to overlay cage over the frame) I created using albumentations library
4) **instance_segmentation** folder contains initial code which could be used for instance segmentation of fighters based on ideas from here https://github.com/ternaus/TernausNetV2; **instance_segmentation/data/processing_functions.py** contains the code for the postprocessing step, which involves watershed transform
5) **labelling_scripts** some scripts I used to check work of freelancers, who were working hard labelling dataset for binary segmentations
