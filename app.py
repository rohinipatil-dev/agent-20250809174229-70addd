import os
import tempfile
import requests
import streamlit as st
from urllib.parse import urlparse
from mimetypes import guess_extension
from openai import OpenAI


# Initialize OpenAI client
client = OpenAI()


# Supported extensions for video/audio that Whisper can handle directly
SUPPORTED_EXTS = {
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".oga", ".mkv", ".mov", ".m4v"
}

# Map common content types to file extensions
CONTENT_TYPE_TO_EXT = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
}


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def infer_extension(url: str, content_type: str | None) -> str:
    # Try to infer extension from URL path
    path_ext = os.path.splitext(urlparse(url).path)[1].lower()
    if path_ext in SUPPORTED_EXTS:
        return path_ext

    # Fallback to content type mapping
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        if ct in CONTENT_TYPE_TO_EXT:
            return CONTENT_TYPE_TO_EXT[ct]
        # Generic guess via mimetypes
        guessed = guess_extension(ct)
        if guessed and guessed in SUPPORTED_EXTS:
            return guessed

    # Default fallback
    return ".mp4"


def download_media(url: str, progress_placeholder) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; StreamlitTranscriber/1.0)"
    }
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        content_type = r.headers.get("Content-Type", "")
        ext = infer_extension(url, content_type)
        # Ensure extension is supported
        if ext not in SUPPORTED_EXTS:
            ext = ".mp4"

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB
        progress_bar = progress_placeholder.progress(0.0, text="Downloading...")
        try:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        progress = min(downloaded / total, 1.0)
                        progress_bar.progress(progress, text=f"Downloading... {downloaded/1_000_000:.1f} MB")
                    else:
                        # Indeterminate progress
                        progress_bar.progress(0.0, text=f"Downloading... {downloaded/1_000_000:.1f} MB")
        finally:
            tmp_file.flush()
            tmp_file.close()
            progress_bar.empty()

        return tmp_file.name


def transcribe_file(file_path: str, language: str | None = None) -> str:
    with open(file_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language if language and language != "Auto-detect" else None
        )
    return transcription.text


def main():
    st.set_page_config(page_title="Video-to-Text Transcriber", page_icon="🎧", layout="centered")
    st.title("Video-to-Text Transcriber")
    st.caption("Paste a direct link to a video/audio file. The app will extract audio and transcribe it using Whisper.")

    url = st.text_input("Video or Audio File URL", placeholder="https://example.com/path/video.mp4")
    language = st.selectbox(
        "Transcription Language (optional)",
        options=["Auto-detect", "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "hi"]
    )

    transcribe_clicked = st.button("Transcribe")

    if transcribe_clicked:
        if not url or not is_valid_url(url):
            st.error("Please enter a valid HTTP/HTTPS URL to a media file.")
            return

        progress_placeholder = st.empty()

        downloaded_path = None
        try:
            with st.spinner("Downloading media..."):
                downloaded_path = download_media(url, progress_placeholder)

            file_size_mb = os.path.getsize(downloaded_path) / 1_000_000
            if file_size_mb > 50:
                st.warning(f"Downloaded file size is {file_size_mb:.1f} MB. Large files may take longer to process.")

            with st.spinner("Transcribing audio with Whisper..."):
                text = transcribe_file(downloaded_path, language=language)
            st.success("Transcription complete.")
            st.text_area("Transcribed Text", value=text, height=300)

            st.download_button(
                label="Download Transcript",
                data=text,
                file_name="transcript.txt",
                mime="text/plain"
            )
        except requests.exceptions.RequestException as re:
            st.error(f"Failed to download media: {str(re)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            progress_placeholder.empty()
            if downloaded_path and os.path.exists(downloaded_path):
                try:
                    os.remove(downloaded_path)
                except Exception:
                    pass


if __name__ == "__main__":
    main()