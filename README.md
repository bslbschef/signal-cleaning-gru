# Wind Speed Denoising with GRU

This project implements a denoising method for wind speed data using a GRU-based model. The data is processed in the frequency domain using FFT, and the GRU model is used to denoise the real and imaginary parts of the FFT result. The model is trained and evaluated on datasets of wind speed signals.

## File Structure

- `data/`: Contains the training and testing data files (`train.mat` and `test.mat`).
- `models/`: Contains the GRU model implementation.
- `preprocess/`: Contains the FFT processing functions.
- `utils/`: Contains utility functions for loading data and plotting results.
- `train.py`: Script to train the model.
- `test.py`: Script to test the model and visualize results.

## Requirements

Install the necessary dependencies with:

```bash
pip install -r requirements.txt
