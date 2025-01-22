# **Advanced Voice Conversion App**

An advanced, state-of-the-art voice conversion application built with **PyTorch**, **Streamlit**, and **Wav2Vec2**. This app leverages pre-trained models for feature extraction and high-quality audio synthesis, allowing users to train on their voice samples and convert audio files into a target voice.

---

## **Features**

- **State-of-the-Art Feature Extraction:**
  - Uses **Wav2Vec2.0** for robust and noise-resilient audio feature extraction.
- **Custom Voice Training:**
  - Train the model with your own voice samples for personalized conversion.
- **High-Quality Audio Reconstruction:**
  - Prepares output compatible with advanced vocoders like **HiFi-GAN** for natural-sounding audio.
- **Simple, Intuitive Interface:**
  - A web-based interface powered by **Streamlit** for ease of use.
- **Efficient and Scalable Design:**
  - Supports GPU acceleration and modularized components for rapid inference and future improvements.

---

## **Getting Started**

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- A compatible GPU (optional but recommended for training and inference)
- Libraries listed in the `requirements.txt` file

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AmandracOP/AVCA.git
   cd AVCA
2. **Make an eniornment**
 
    ```bash
    python -m venv navya
    source navya/bin/activate #for linux
    navya/Scripts/activate #for windows 

3. **Install all the dependencies**

    ```bash
    pip install -r requirements.txt

4. **Run Streamlit App**
    ```bash
    streamlit run src/webapp/app.py  

## Usage
1. **Train the Model**

    Upload multiple audio files of your voice (.wav or .mp3).
    Click on the "Train Model" button.
    Wait for the training process to complete (takes a few minutes depending on your system).

2. **Convert Audio**

    Upload an audio file you want to convert.
    Click on the "Convert" button.
    Download the converted audio.    
## Technical Details
### Architecture

    1. Feature Extraction: Uses Wav2Vec 2.0 for audio embedding.
    Voice Conversion Model:
        A custom feedforward neural network trained to transform voice features.
        Compatible with pre-trained generative models (e.g., HiFi-GAN).
    2. Audio Reconstruction:
        Uses vocoders like HiFi-GAN or WaveGlow for converting embeddings to waveforms.

### Optimization

    Mixed precision training using torch.cuda.amp.
    GPU acceleration for faster training and inference.
    Modular design for easy integration of future models.

## License

**This project is licensed under the MIT License. See the LICENSE file for details.**