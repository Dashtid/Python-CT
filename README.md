# CT Image Processing, Segmentation, and Classification

This repository provides tools for processing, segmenting, evaluating, and classifying CT images, with a focus on atlas-based segmentation and random forest classification. The code is modular and uses Python with SimpleITK.

## Features

- **Pre-processing:** Load and prepare CT images and masks.
- **Registration:** Linear and non-linear (Demons) registration using SimpleITK.
- **Atlas-based Segmentation:** Majority voting from registered masks.
- **Evaluation:** Computes Dice, Jaccard, and Hausdorff metrics.
- **Classification:** Random forest classifier to identify slices containing the pubic symphysis.
- **Visualization:** Plotting of images and segmentation overlays.

## Directory Structure

.
├── main.py # Main script for running the pipeline
├── functions.py # Core functions for registration, segmentation, evaluation, and classification
├── requirements.txt # Python dependencies  
├── README.md
└── .gitignore

## Getting Started

### Prerequisites

- Python 3.8+
- See [requirements.txt](requirements.txt) for dependencies

### Installation

```sh
git clone https://github.com/yourusername/your-repo.git
cd Python-CT
pip install -r requirements.txt
```

### Usage

1. Place your CT images and masks in the appropriate folders (see paths in `Main.py`).
2. Run the main script:
   ```sh
   python Main.py
   ```
3. The script will:
   - Load images and masks
   - Perform registration and segmentation
   - Evaluate segmentation quality
   - Train and apply a classifier
   - Display plots of results

### File Descriptions

- **[Main.py](Main.py):** Entry point; orchestrates the workflow.
- **[functions.py](functions.py):** Implements registration, segmentation, evaluation, and classification functions.
- **[requirements.txt](requirements.txt):** Lists required Python packages.

### Example Output

- Dice, Jaccard, and Hausdorff metrics printed to console.
- Plots showing reference images, segmentations, and overlays.
- Slice indices with highest probability for pubic symphysis.

## Notes

- Image and mask file paths are hardcoded in `Main.py`. Adjust as needed for your dataset.
- The code assumes images are in NIfTI format (`.nii.gz`).
- For more advanced usage or batch processing, consider extending the scripts or modularizing further.

## License

MIT License

---

_Maintained by David Dashti. For questions or suggestions, please open an issue or contact me._
