import streamlit as st
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import soundfile as sf
import tempfile
from pathlib import Path
from pydub import AudioSegment


class VoiceConverter:
    def __init__(self):
        self.sample_rate = 16000
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        self.feature_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
        self.feature_extractor.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor.to(self.device)
        self.model = None

    def preprocess_audio(self, audio_path):
        """Preprocess audio into embeddings using Wav2Vec2."""
        waveform, sr = torchaudio.load(audio_path)
        waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

        with torch.no_grad():
            inputs = self.processor(waveform, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
            embeddings = self.feature_extractor(inputs.input_values.to(self.device)).last_hidden_state
        return embeddings

    def train_model(self, training_audio_paths):
        """Finetune the voice conversion model."""
        st.write("Extracting features for training data...")
        training_data = [self.preprocess_audio(path) for path in training_audio_paths]

        # Placeholder: Pre-trained model usage (e.g., fine-tune HiFi-GAN or other state-of-the-art models)
        # Model architecture can be loaded here, such as Tacotron or a Transformer-based model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(training_data[0].size(-1), 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, training_data[0].size(-1))
        ).to(self.device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        st.write("Training the model...")
        for epoch in range(5):  # Minimal epochs for demonstration
            epoch_loss = 0
            for data in training_data:
                optimizer.zero_grad()
                data = data.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, data)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            st.write(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        st.success("Model training completed!")

    def convert_voice(self, input_audio_path, output_path):
        """Convert voice using the trained model."""
        if not self.model:
            raise ValueError("Model needs to be trained first!")

        input_embeddings = self.preprocess_audio(input_audio_path)
        with torch.no_grad():
            converted_embeddings = self.model(input_embeddings.to(self.device))

        # Use HiFi-GAN or WaveGlow for vocoding (reconstruct audio from embeddings)
        st.warning("This code assumes a vocoder like HiFi-GAN is integrated for reconstruction.")
        waveform = torch.randn(1, int(self.sample_rate * 5))  # Dummy waveform for testing purposes
        sf.write(output_path, waveform.cpu().numpy(), self.sample_rate)
        return output_path


def convert_to_wav(file):
    """Convert any audio file to .wav."""
    audio = AudioSegment.from_file(file)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_file.name, format="wav")
    return temp_file.name


def main():
    st.title("Advanced Voice Conversion App")

    if 'voice_converter' not in st.session_state:
        st.session_state.voice_converter = VoiceConverter()

    st.header("1. Train with Your Voice")
    training_files = st.file_uploader("Upload voice samples (WAV/MP3)", type=['wav', 'mp3'], accept_multiple_files=True)

    if training_files and st.button("Train Model"):
        temp_paths = [convert_to_wav(file) for file in training_files]

        try:
            st.session_state.voice_converter.train_model(temp_paths)
        finally:
            for path in temp_paths:
                Path(path).unlink()

    st.header("2. Convert Audio")
    input_file = st.file_uploader("Upload audio file (WAV/MP3)", type=['wav', 'mp3'])

    if input_file and st.button("Convert"):
        if not st.session_state.voice_converter.model:
            st.error("Train the model first!")
            return

        input_path = convert_to_wav(input_file)
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        try:
            st.session_state.voice_converter.convert_voice(input_path, output_path)

            with open(output_path, "rb") as f:
                st.download_button("Download Converted Audio", f, "converted_audio.wav", "audio/wav")
        finally:
            Path(input_path).unlink()
            Path(output_path).unlink()


if __name__ == "__main__":
    main()
