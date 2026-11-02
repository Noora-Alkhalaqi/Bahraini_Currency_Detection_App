# Bahraini Currency Detector

A Streamlit app that finds Bahraini banknotes and coins in an image or camera feed using a custom-trained YOLO26 model. For images and photos, it also adds up the total value in Bahraini dinars (BD).

**Try it live:** [Bahraini Currency Detection App](https://bahrainicurrencydetectionapp.streamlit.app/)

## Features

- **Upload images:** run detection on one or more JPG, JPEG, PNG or WebP files.
- **Take a photo:** capture a single image with your camera.
- **Live camera:** see detections drawn on a real-time video stream.
- **Value summary:** for uploads and photos, see the estimated total plus the count and subtotal for each denomination.

## Supported denominations

The model detects 10 classes:

| Type | Denominations |
| --- | --- |
| Coins | 5 fils, 10 fils, 25 fils, 50 fils, 100 fils |
| Banknotes | 0.5 BD, 1 BD, 5 BD, 10 BD, 20 BD |

1 BD equals 1000 fils.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Noora-Alkhalaqi/Bahraini_Currency_Detection_App.git
cd Bahraini_Currency_Detection_App
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Streamlit prints a local URL, usually http://localhost:8501. The app expects `best.pt` to be in the same folder as `app.py`.

Browsers only allow camera access on `localhost` or over HTTPS, so the camera tabs need HTTPS if you deploy the app.

## Model

| | |
| --- | --- |
| Base model | `yolo26s.pt` (pretrained YOLO26s, fine-tuned on this dataset) |
| Framework | Ultralytics 8.4.198 |
| Dataset | Custom dataset from Roboflow (`bahraini-currency-1ka60`, version 1) |
| Images | 362 training, 11 validation |
| Image size | 640 |
| Epochs | 5 |
| Training hardware | Google Colab, Tesla T4 GPU |

### Validation results

| Metric | Score |
| --- | --- |
| Precision | 0.850 |
| Recall | 0.837 |
| mAP50 | 0.922 |
| mAP50-95 | 0.859 |

These scores are a rough guide only. They come from just 11 images containing 12 currency instances, and only 6 of the 10 classes appear in that set.

## Limitations

- **Untested classes:** the 5, 10, 25 and 100 fils coins have no validation metrics, because they don't appear in the validation set.
- **Weakest validated classes:** 20 BD (recall 0.42) and 0.5 BD (recall 0.60) are the most likely to be missed.
- **Short training:** the model was trained for only 5 epochs on a small dataset, so it may not hold up in unfamiliar lighting, angles or backgrounds.
- **Estimated totals:** the total comes straight from the detections, so a missed or misclassified item changes it. Don't rely on it for real cash handling.
- **Live mode:** the live camera tab draws detections but doesn't show a total.

## Project structure

```text
.
├── app.py                    # Streamlit app
├── best.pt                   # Trained YOLO26s weights
├── bahraini_currency.ipynb   # Training notebook (Google Colab)
├── currencyCamera.ipynb      # Webcam detection test in Jupyter
├── requirements.txt          # Python dependencies
└── README.md
```

## Built with

- [Ultralytics YOLO](https://docs.ultralytics.com/) for object detection
- [Roboflow](https://roboflow.com/) for dataset management
- [Streamlit](https://streamlit.io/) and [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) for the web app and live video
- [OpenCV](https://opencv.org/) and [Pillow](https://python-pillow.org/) for image processing
