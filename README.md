# AI Style Transfer Studio

A powerful Stable Diffusion based application that allows you to transform your text prompts into artistic masterpieces using various trained style models.

## Features

- **Multiple Artistic Styles**: Transform your prompts using various pre-trained styles:
  - Dhoni Style (Sports/Action)
  - Mickey Mouse Style (Cartoon/Whimsical)
  - Balloon Style (Festive/Celebratory)
  - Lion King Style (Majestic/Animal)
  - Rose Flower Style (Floral/Romantic)

- **Color Enhancement**: Advanced color processing technology that maximizes color channel separation for more vibrant and striking images

- **User-Friendly Interface**: Clean, modern web interface built with Streamlit

- **Style Training**: Custom script for training new style embeddings using your own image datasets

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have the following system requirements:
   - Python 3.7 or higher
   - CUDA-compatible GPU (recommended) or CPU
   - At least 8GB RAM (16GB recommended)

## Usage

### Running the Application

1. Start the web interface:
   ```bash
   python src/app.py
   ```

2. Open your web browser and navigate to the displayed URL

3. Enter your text prompt and select a style

4. Click "Generate Artwork" to create your masterpiece

### Training New Styles

1. Prepare your training images:
   - Place your style reference images in `training_images/<style_name>/`
   - Supported formats: JPG, PNG
   - Recommended: 3-5 high-quality images per style

2. Run the training script:
   ```bash
   python src/train_style.py <style_name> --steps 3000 --lr 1e-4
   ```

## Project Structure

```
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── train_style.py         # Style training script
│   └── utils/
│       ├── style_generator.py # Core style transfer logic
│       └── ui_components.py   # UI component definitions
├── training_images/           # Training image datasets
└── requirements.txt          # Project dependencies
```

## Technical Details

- Built on Stable Diffusion v1.5
- Uses textual inversion for style embedding
- Implements custom color enhancement using distance loss
- Supports both CPU and CUDA acceleration

## Google Colab Logs

```bash
!python src/train_style.py dhoni --steps 1000 --lr 1e-4
```


```Log 
2025-02-26 08:23:28.807019: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1740558208.828141   65129 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1740558208.834586   65129 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered

Training dhoni style...
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
Step 0: Loss 0.23166896402835846
```

The above command generates the embedding file "dhoni.bin" in "style_embeddings" folder. 
All embedding files are generated based on the above command, by providing the desired style name.
Generated bin files are used in the HuggingFace App and not uploaded with this repo do avoid size constraints of 100 MB on GitHub.
