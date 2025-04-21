# ProjectSEM1
An A project of SEM1 ( DL – Based detection and documentation in MRI scans ) 

Main Idea of the Project:

The project aims to develop an artificial intelligence system for detecting and documenting brain tumors in MRI scans. It consists of four main phases:

* Data Processing: Organizing and preprocessing MRI images from a Kaggle dataset, applying image processing techniques such as resizing, normalization, and data augmentation.
Segmentation and Detection: Utilizing a U-Net model based on Convolutional Neural Networks (CNN) to segment images and accurately identify tumor locations.

* Feature Extraction: Employing the PyRadiomics library to extract quantitative features from regions of interest (ROI), such as intensity, contrast, and texture, and storing them in a CSV file.

* Documentation: Translating extracted features into a comprehensible medical report using simplified medical terminology, presented through a graphical user interface (GUI) or API to assist doctors and patients in understanding the condition.
The system seeks to enhance the accuracy of detecting tumors (glioma, meningioma, and pituitary) and provide visual reports to facilitate clinical decision-making, with future plans to include additional tumor types and integrate other medical records, such as blood tests, to support diagnosis.
