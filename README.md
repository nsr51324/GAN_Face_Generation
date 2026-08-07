# GAN Face Generation

A professional Streamlit-based application for generating human face images using a trained Generative Adversarial Network (GAN) model.

This project allows users to load a pretrained generator and create realistic-looking synthetic face images from random noise vectors with a simple and interactive web interface.

---

## 1. Project Overview

This repository contains:

- A Streamlit web app for face generation
- A pretrained generator model
- Dataset metadata for the CelebA dataset
- A Jupyter notebook with the training and experimentation workflow

The app is designed to be easy to run locally and provides a simple interface for generating multiple face images at once.

---

## 2. Features

- Generate synthetic human face images from random latent vectors
- Adjustable number of generated images
- Optional fixed-seed generation for reproducible results
- Simple Streamlit UI with sidebar controls
- Pretrained model support from the local project folder

---

## 3. Project Structure

```text
GAN_Face_Generation/
│
├── app.py                    # Main Streamlit application
├── requirements.txt         # Python dependencies
├── data/                     # CelebA metadata files
│   ├── list_attr_celeba.csv
│   ├── list_bbox_celeba.csv
│   ├── list_eval_partition.csv
│   └── list_landmarks_align_celeba.csv
├── models/                   # Trained GAN weights
│   ├── discriminator_final.pth
│   ├── full_checkpoint_final.pth
│   ├── generator_final.pth
│   └── training_history.json
├── notebook/                 # Training notebook
│   └── GAN_Face_Generation (1).ipynb
└── .venv/                    # Local virtual environment (created when needed)
```

---

## 4. Technologies Used

- Python
- PyTorch
- Streamlit
- Torchvision
- NumPy
- Pillow
- Jupyter Notebook

---

## 5. Requirements

The project requires Python 3.10+ and the packages listed in [requirements.txt](requirements.txt).

### Main dependencies

- streamlit
- torch
- torchvision
- numpy
- Pillow

---

## 6. Installation

### Option 1: Use the provided virtual environment

If the virtual environment already exists in the project folder, activate it:

```powershell
cd C:\Users\LOQ\Desktop\GAN_Face_Generation
.venv\Scripts\Activate.ps1
```

### Option 2: Create a new virtual environment

```powershell
cd C:\Users\LOQ\Desktop\GAN_Face_Generation
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

> If you are using Windows and PowerShell blocks script execution, run the following once:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

---

## 7. Running the Application

Run the Streamlit app using:

```powershell
cd C:\Users\LOQ\Desktop\GAN_Face_Generation
python -m streamlit run app.py
```

Or, if you want to use the project virtual environment explicitly:

```powershell
cd C:\Users\LOQ\Desktop\GAN_Face_Generation
.venv\Scripts\python.exe -m streamlit run app.py
```

After the app starts, open your browser and go to:

```text
http://localhost:8501
```

---

## 8. How to Use the App

1. Open the app in the browser.
2. In the sidebar, choose the number of images to generate.
3. Optionally enable a fixed seed for reproducible output.
4. Click the Generate faces button.
5. The app will load the pretrained generator and display the generated images.

---

## 9. Model Files

The pretrained model files are stored in the [models](models) folder:

- [models/generator_final.pth](models/generator_final.pth) — the trained generator weights used by the app
- [models/discriminator_final.pth](models/discriminator_final.pth) — discriminator weights from training
- [models/full_checkpoint_final.pth](models/full_checkpoint_final.pth) — full training checkpoint
- [models/training_history.json](models/training_history.json) — training metrics/history

The app loads the generator model by default from the models folder.

---

## 10. Dataset Information

The project uses metadata related to the CelebA dataset, stored under [data](data):

- identity/attribute labels
- bounding boxes
- partition information
- facial landmarks

These files support the training and analysis workflow but are not required for the simple generation UI.

---

## 11. Notebook

The notebook in [notebook/GAN_Face_Generation (1).ipynb](notebook/GAN_Face_Generation%20(1).ipynb) contains the experimentation and training workflow used to develop the model.

It is useful for:

- understanding model architecture
- reviewing training logic
- experimenting with different GAN settings

---

## 12. Troubleshooting

### Problem: PyTorch fails to import

This can happen when the installed PyTorch build is not compatible with your current environment.

Try reinstalling dependencies in the project virtual environment:

```powershell
cd C:\Users\LOQ\Desktop\GAN_Face_Generation
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Problem: Streamlit does not open in the browser

Make sure the app is running and browse to:

```text
http://localhost:8501
```

If the port is busy, Streamlit may choose another port. Check the terminal output for the correct URL.

### Problem: Model file not found

The app expects the generator file to exist at:

```text
models/generator_final.pth
```

If the file is missing or moved, update the path in the sidebar input.

---

## 13. Notes and Limitations

- This is a demonstration project focused on image generation.
- The generated outputs depend on the pretrained generator and random noise input.
- The visual quality may vary depending on the model weights and environment.

---

## 14. Summary

This project provides a clean and practical example of:

- building a GAN-based face generation application
- deploying it through Streamlit
- using pretrained PyTorch weights for inference
- creating an interactive UI for image generation

If you want, I can also create a second version of this README in Arabic only, or add a screenshot section and a "How it works" diagram for a more polished presentation.
