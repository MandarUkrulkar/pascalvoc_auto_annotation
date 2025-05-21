# YOLOv8 Annotation Generator

This script performs object detection on a folder of images using a trained YOLOv8 model and generates Pascal VOC-style XML annotation files for each image.

## 🧠 Features

- Uses YOLOv8 (`Ultralytics`) for inference
- Automatically maps class indices to names from the model itself
- Generates XML annotations in Pascal VOC format
- Option to filter detections based on confidence threshold and minimum number of detections

---

## 🗂️ Directory Structure

```
project/
├── script.py
├── requirements.txt
├── README.md
└── outputs/
    └── *.xml
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

You also need a compatible PyTorch version for YOLOv8. Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) to install it based on your CUDA version.

---

## 🚀 Usage

Run the script:

```bash
python script.py
```

You will be prompted to input:
- Path to the directory containing input images
- Output directory to save XML files
- Minimum number of detections per image
- Confidence threshold for filtering detections

---

## 📁 Example

```
Enter the directory path containing images: ./images
Enter the output directory path: ./annotations
Enter the minimum number of detections required in an image: 1
Enter the confidence threshold for detections (0-1): 0.4
```

Each image will be processed, and an XML file with bounding box annotations will be saved in the output directory.

---

## 📝 Notes

- Supported image formats: `.jpg`, `.png`
- All class names are read directly from the YOLOv8 model (`.pt` file)
- Pascal VOC format is compatible with many ML workflows and annotation tools

---

## 🧑‍💻 Author

Developed by Mandar Ukrulkar 
Project purpose:  classification and annotation generation

---

## 📜 License

MIT License (or your chosen license)
