import os
import io
from flask import Flask, render_template, request, jsonify
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Single global client (reuses connection for every request)
speech_client = speech.SpeechClient()

def allowed_file(filename: str) -> bool:
    """Allow common audio formats."""
    ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "m4a"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """Handle audio upload and return diarized transcript as JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # Save uploaded file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        segments = process_audio_with_diarization(filepath)
        return jsonify({"segments": segments})
    except Exception as e:
        # For debugging; in a real deployment you would log this.
        return jsonify({"error": str(e)}), 500

def process_audio_with_diarization(input_path: str):
    """
    1. Normalize audio (mono, 16 kHz) so the speech model can handle it robustly.
    2. Call the cloud speech service with diarization enabled and allow automatic detection of up to 20 speakers.
    3. Group words by speaker and return structured segments.
    """
    # --- 1) Convert/normalize audio using pydub ---
    sound = AudioSegment.from_file(input_path)
    sound = sound.set_channels(1).set_frame_rate(16000)
    processed_path = os.path.join(UPLOAD_FOLDER, "processed.wav")
    sound.export(processed_path, format="wav")

    duration_sec = float(sound.duration_seconds)

    # --- 2) Read audio bytes ---
    with io.open(processed_path, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)

    # Use SpeakerDiarizationConfig to let the API guess up to 20 speakers
    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=1,
        max_speaker_count=20,
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
        diarization_config=diarization_config,
        model="latest_long",
    )

    # Use long_running_recognize for longer files so bigger uploads still work
    if duration_sec > 55:  # ~1 minute threshold
        operation = speech_client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=600)  # up to 10 minutes
    else:
        # Use a faster model for shorter clips
        config.model = "latest_short"
        response = speech_client.recognize(config=config, audio=audio)

    if not response.results:
        return []

    # diarized word info is in the last result
    result = response.results[-1]
    alternative = result.alternatives[0]
    words = alternative.words

    segments = []
    if not words:
        # Fallback: one block, no speaker labels
        segments.append(
            {
                "speaker": "unknown",
                "start": 0.0,
                "end": duration_sec,
                "text": alternative.transcript.strip(),
            }
        )
        return segments

    # --- 3) Group consecutive words by speaker_tag ---
    current_speaker = words[0].speaker_tag
    current_start = words[0].start_time.total_seconds()
    current_words = []

    for idx, w in enumerate(words):
        speaker = w.speaker_tag
        word_text = w.word

        if speaker != current_speaker:
            # close previous segment using previous word's end time
            last_word = words[idx - 1]
            end_time = last_word.end_time.total_seconds()
            segments.append(
                {
                    "speaker": f"Speaker {current_speaker}",
                    "start": round(current_start, 2),
                    "end": round(end_time, 2),
                    "text": " ".join(current_words).strip(),
                }
            )
            # start new segment
            current_speaker = speaker
            current_start = w.start_time.total_seconds()
            current_words = [word_text]
        else:
            current_words.append(word_text)

    # close last segment
    last_word = words[-1]
    last_end = last_word.end_time.total_seconds()
    segments.append(
        {
            "speaker": f"Speaker {current_speaker}",
            "start": round(current_start, 2),
            "end": round(last_end, 2),
            "text": " ".join(current_words).strip(),
        }
    )

    return segments

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)