import streamlit as st
import torch
import torchaudio
from torchaudio.transforms import Resample
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
        try:
            waveform, sr = torchaudio.load(audio_path)  # Use torchaudio directly
            waveform = Resample(sr, self.sample_rate)(waveform)
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

            waveform = waveform.squeeze()  # Shape: [sequence_length]
            waveform = waveform.unsqueeze(0)  # Shape: [1, sequence_length]

            with torch.no_grad():
                inputs = self.processor(
                    waveform, sampling_rate=self.sample_rate, return_tensors="pt", padding=True
                )
                embeddings = self.feature_extractor(inputs.input_values.to(self.device)).last_hidden_state
            return embeddings
        except Exception as e:
            st.error(f"Error in preprocessing audio: {e}")
            return None

    def train_model(self, training_audio_paths, epochs=5, learning_rate=0.0001):
        """Finetune the voice conversion model."""
        st.write("Extracting features for training data...")
        training_data = [self.preprocess_audio(path) for path in training_audio_paths]
        training_data = [data for data in training_data if data is not None]  # Filter out errors

        if not training_data:
            st.error("No valid training data available.")
            return

        input_dim = training_data[0].size(-1)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, input_dim)
        ).to(self.device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        st.write("Training the model...")
        for epoch in range(epochs):
            epoch_loss = 0
            progress = st.progress(0)
            for idx, data in enumerate(training_data):
                optimizer.zero_grad()
                data = data.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, data)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                progress.progress((idx + 1) / len(training_data))
            st.write(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        st.success("Model training completed!")

    def convert_voice(self, input_audio_path, output_path):
        """Convert voice using the trained model."""
        if not self.model:
            st.error("Model needs to be trained first!")
            return None

        input_embeddings = self.preprocess_audio(input_audio_path)
        if input_embeddings is None:
            st.error("Error in preprocessing input audio.")
            return None

        with torch.no_grad():
            converted_embeddings = self.model(input_embeddings.to(self.device))

        # Placeholder for vocoder integration (replace with actual vocoder usage)
        st.warning("This code assumes a vocoder like HiFi-GAN is integrated for reconstruction.")
        waveform = torch.randn(1, int(self.sample_rate * 5))  # Dummy waveform for testing purposes
        sf.write(output_path, waveform.cpu().numpy(), self.sample_rate)
        return output_path

    def save_model(self, path="voice_model.pth"):
        if self.model:
            torch.save(self.model.state_dict(), path)
            st.success(f"Model saved to {path}")
        else:
            st.error("No trained model to save.")

    def load_model(self, path="voice_model.pth"):
        try:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 512)
            ).to(self.device)
            self.model.load_state_dict(torch.load(path))
            st.success("Model loaded successfully.")
        except Exception as e:
            st.error(f"Error loading model: {e}")


def convert_to_wav(file):
    """Convert any audio file to .wav."""
    try:
        audio = AudioSegment.from_file(file)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(temp_file.name, format="wav")
        return temp_file.name
    except Exception as e:
        st.error(f"Error converting to WAV: {e}")
        return None


def main():
    st.title("Advanced Voice Conversion App")

    if 'voice_converter' not in st.session_state:
        st.session_state.voice_converter = VoiceConverter()

    vc = st.session_state.voice_converter

    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Choose Action", ["Train Model", "Convert Audio", "Save Model", "Load Model"])

    if option == "Train Model":
        st.header("1. Train with Your Voice")
        training_files = st.file_uploader("Upload voice samples (WAV/MP3)", type=['wav', 'mp3'], accept_multiple_files=True)

        if training_files and st.button("Train Model"):
            temp_paths = [convert_to_wav(file) for file in training_files]
            temp_paths = [path for path in temp_paths if path]

            try:
                vc.train_model(temp_paths)
            finally:
                for path in temp_paths:
                    if Path(path).exists():
                        Path(path).unlink()

    elif option == "Convert Audio":
        st.header("2. Convert Audio")
        input_file = st.file_uploader("Upload audio file (WAV/MP3)", type=['wav', 'mp3'])

        if input_file and st.button("Convert"):
            input_path = convert_to_wav(input_file)
            output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

            if input_path:
                try:
                    converted_file = vc.convert_voice(input_path, output_path)

                    if converted_file:
                        st.audio(converted_file, format="audio/wav")

                        with open(converted_file, "rb") as f:
                            st.download_button("Download Converted Audio", f, "converted_audio.wav", "audio/wav")
                finally:
                    for path in [input_path, output_path]:
                        if Path(path).exists():
                            Path(path).unlink()

    elif option == "Save Model":
        model_path = st.text_input("Enter path to save model", "voice_model.pth")
        if st.button("Save Model"):
            vc.save_model(model_path)

    elif option == "Load Model":
        model_path = st.text_input("Enter path to load model", "voice_model.pth")
        if st.button("Load Model"):
            vc.load_model(model_path)


if __name__ == "__main__":
    main()
