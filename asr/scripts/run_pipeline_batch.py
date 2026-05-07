import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.settings.config import Config
from src.audio.extractor import AudioExtractor


# ============================================================
# Whisper clients
# ============================================================

class OpenAIWhisperClient:
    """
    Whisper client using the OpenAI API.
    Returns a normalized dict:
    {
        "text": str,
        "segments": [{"start": float, "end": float, "text": str}, ...],
        "words": [{"start": float, "end": float, "word": str}, ...]
    }
    """
    def __init__(self, config: Config):
        import openai

        self.config = config
        self.client = openai.OpenAI(api_key=config.openai_api_key)
        self.logger = logging.getLogger(__name__)

    def transcribe_audio(self, audio_path: Path, **kwargs) -> Dict[str, Any]:
        start_time = time.time()

        try:
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                    **kwargs,
                )

            duration = time.time() - start_time
            self.logger.info(
                f"[OpenAI API] Transcription succeeded for {audio_path} in {duration:.2f} seconds."
            )

            segments = []
            if hasattr(response, "segments") and response.segments:
                for seg in response.segments:
                    segments.append(
                        {
                            "start": float(seg.start),
                            "end": float(seg.end),
                            "text": str(seg.text),
                        }
                    )

            words = []
            if hasattr(response, "words") and response.words:
                for w in response.words:
                    words.append(
                        {
                            "start": float(w.start),
                            "end": float(w.end),
                            "word": str(w.word),
                        }
                    )

            return {
                "text": str(response.text),
                "segments": segments,
                "words": words,
            }

        except TimeoutError as e:
            duration = time.time() - start_time
            self.logger.error(
                f"[OpenAI API] Timeout for {audio_path} after {duration:.2f}s: {e}"
            )
            raise
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"[OpenAI API] Transcription failed for {audio_path} after {duration:.2f}s: {e}"
            )
            raise

    def transcribe_chunks(self, audio_chunks: List[Path]) -> Dict[str, Any]:
        return _transcribe_chunks_shared(self, audio_chunks)


class LocalWhisperClient:
    """
    Local Whisper client using the `openai-whisper` package.
    Returns the same normalized structure as the API client.
    """
    def __init__(self, config: Config):
        import whisper
        import torch

        self.config = config
        self.logger = logging.getLogger(__name__)

        model_name = config.whisper_model_name
        self.language = config.whisper_language

        self.device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"

        self.logger.info(f"Loading local Whisper model: {model_name} on {self.device}")
        self.model = whisper.load_model(model_name, device=self.device)
        # self.logger.info(f"Loading local Whisper model: {model_name}")
        # self.model = whisper.load_model(model_name)

    def transcribe_audio(self, audio_path: Path, **kwargs) -> Dict[str, Any]:
        start_time = time.time()

        try:
            result = self.model.transcribe(
                str(audio_path),
                word_timestamps=True,
                language=self.language,
                **kwargs,
            )

            duration = time.time() - start_time
            self.logger.info(
                f"[Local Whisper] Transcription succeeded for {audio_path} in {duration:.2f} seconds."
            )

            segments = []
            words = []

            for seg in result.get("segments", []):
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", 0.0))
                seg_text = str(seg.get("text", ""))

                segments.append(
                    {
                        "start": seg_start,
                        "end": seg_end,
                        "text": seg_text,
                    }
                )

                for w in seg.get("words", []) or []:
                    w_start = w.get("start", None)
                    w_end = w.get("end", None)
                    w_word = w.get("word", "")

                    if w_start is not None and w_end is not None:
                        words.append(
                            {
                                "start": float(w_start),
                                "end": float(w_end),
                                "word": str(w_word),
                            }
                        )

            return {
                "text": str(result.get("text", "")),
                "segments": segments,
                "words": words,
            }

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"[Local Whisper] Transcription failed for {audio_path} after {duration:.2f}s: {e}"
            )
            raise

    def transcribe_chunks(self, audio_chunks: List[Path]) -> Dict[str, Any]:
        return _transcribe_chunks_shared(self, audio_chunks)


def _transcribe_chunks_shared(client_obj, audio_chunks: List[Path]) -> Dict[str, Any]:
    """
    Shared chunk transcription/merge logic for both API and local clients.
    """
    logger = logging.getLogger(__name__)

    if not audio_chunks:
        return {"text": "", "segments": [], "words": []}

    if len(audio_chunks) == 1:
        return client_obj.transcribe_audio(audio_chunks[0])

    logger.info(f"Transcribing {len(audio_chunks)} audio chunks...")

    all_texts = []
    all_segments = []
    all_words = []
    cumulative_offset = 0.0
    chunks_transcribed = 0

    for i, chunk_path in enumerate(audio_chunks):
        try:
            logger.info(f"Transcribing chunk {i+1}/{len(audio_chunks)}: {chunk_path.name}")
            result = client_obj.transcribe_audio(chunk_path)

            all_texts.append(result.get("text", ""))
            chunks_transcribed += 1

            chunk_segments = result.get("segments", [])
            for seg in chunk_segments:
                all_segments.append(
                    {
                        "start": float(seg["start"]) + cumulative_offset,
                        "end": float(seg["end"]) + cumulative_offset,
                        "text": str(seg["text"]),
                    }
                )

            chunk_words = result.get("words", [])
            for w in chunk_words:
                all_words.append(
                    {
                        "start": float(w["start"]) + cumulative_offset,
                        "end": float(w["end"]) + cumulative_offset,
                        "word": str(w["word"]),
                    }
                )

            actual_chunk_duration = 0.0
            if chunk_segments:
                actual_chunk_duration = float(chunk_segments[-1]["end"])
            elif chunk_words:
                actual_chunk_duration = float(chunk_words[-1]["end"])

            if actual_chunk_duration <= 0:
                logger.warning(
                    f"No timing data found for chunk {chunk_path.name}; using fallback 600s"
                )
                actual_chunk_duration = 600.0

            cumulative_offset += actual_chunk_duration

        except Exception as e:
            logger.error(f"Failed to transcribe chunk {i+1} ({chunk_path.name}): {e}")
            all_texts.append(f"[ERROR: Failed to transcribe chunk {i+1}]")
            cumulative_offset += 600.0

    combined_text = " ".join(t.strip() for t in all_texts if t.strip())

    return {
        "text": combined_text,
        "segments": all_segments,
        "words": all_words,
        "chunk_count": len(audio_chunks),
        "chunks_transcribed": chunks_transcribed,
    }


# ============================================================
# Pipeline
# ============================================================

class SurgeryTranscriptionPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.audio_extractor = AudioExtractor(config)
        self.logger = logging.getLogger(__name__)

        whisper_backend = config.whisper_backend.lower()
        if whisper_backend == "local":
            self.whisper_client = LocalWhisperClient(config)
            self.logger.info("Using local Whisper backend")
        else:
            self.whisper_client = OpenAIWhisperClient(config)
            self.logger.info("Using OpenAI Whisper API backend")

    def process_video(self, video_path: Path) -> Dict[str, Any]:
        start_time = time.time()
        timing_info = {}

        # 1. Extract audio
        audio_start = time.time()
        audio_path = self.audio_extractor.extract_audio(video_path)
        audio_time = time.time() - audio_start
        timing_info["audio_extraction"] = audio_time
        print(f"Audio extraction completed in {audio_time:.2f} seconds")

        # 2. Chunk if needed
        chunk_start = time.time()
        audio_chunks = self.audio_extractor.chunk_audio_if_needed(audio_path)
        chunk_time = time.time() - chunk_start
        timing_info["audio_chunking"] = chunk_time
        print(f"Audio chunking completed in {chunk_time:.2f} seconds ({len(audio_chunks)} chunks)")

        # 3. Transcribe
        transcription_start = time.time()
        if len(audio_chunks) == 1:
            transcript = self.whisper_client.transcribe_audio(audio_chunks[0])
        else:
            transcript = self.whisper_client.transcribe_chunks(audio_chunks)
        transcription_time = time.time() - transcription_start
        timing_info["transcription"] = transcription_time
        print(f"Transcription completed in {transcription_time:.2f} seconds")

        # # 4. Save outputs
        # save_start = time.time()
        # output_dir = Path(self.config.output_dir)
        # output_dir.mkdir(parents=True, exist_ok=True)

        # 4. Save outputs
        save_start = time.time()

        root_output_dir = Path(self.config.output_dir)
        video_folder_name = self._safe_folder_name(video_path.stem)
        output_dir = root_output_dir / "narrations" / video_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = video_folder_name
        # Plain text transcript
        transcript_path = output_dir / f"{base_name}_transcript.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(
                transcript["text"]
                if isinstance(transcript, dict) and "text" in transcript
                else str(transcript)
            )

        # Segment-level timestamped transcript
        try:
            if isinstance(transcript, dict) and transcript.get("segments"):
                timestamped_lines = []
                for seg in transcript["segments"]:
                    start_str = self._format_timestamp(seg["start"])
                    end_str = self._format_timestamp(seg["end"])
                    seg_text = seg["text"].strip()
                    timestamped_lines.append(f"[{start_str} --> {end_str}] {seg_text}")

                timestamped_path = output_dir / f"{base_name}_transcript_timestamped.txt"
                with open(timestamped_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(timestamped_lines))
            else:
                print("Warning: No segment timestamps found in transcript")
        except Exception as e:
            print(f"Warning: Failed to save segment timestamped transcript: {e}")

        # Word-level timestamped transcript
        try:
            if isinstance(transcript, dict) and transcript.get("words"):
                word_path = output_dir / f"{base_name}_transcript_words.txt"
                with open(word_path, "w", encoding="utf-8") as f:
                    for w in transcript["words"]:
                        start_str = self._format_timestamp(w["start"])
                        end_str = self._format_timestamp(w["end"])
                        f.write(f"[{start_str} --> {end_str}] {w['word']}\n")
            else:
                print("Warning: No word timestamps found in transcript")
        except Exception as e:
            print(f"Warning: Failed to save word-level transcript: {e}")

        # Sentence-level transcript from words
        try:
            if isinstance(transcript, dict) and transcript.get("words"):
                sentence_items = self._build_sentences_from_words(transcript["words"])

                sentence_path = output_dir / f"{base_name}_transcript_sentences.txt"
                with open(sentence_path, "w", encoding="utf-8") as f:
                    for s in sentence_items:
                        start_str = self._format_timestamp(s["start"])
                        end_str = self._format_timestamp(s["end"])
                        f.write(f"[{start_str} --> {end_str}] {s['text']}\n")
            else:
                print("Warning: No word timestamps found for sentence-level transcript")
        except Exception as e:
            print(f"Warning: Failed to save sentence-level transcript: {e}")

        # Detailed JSON
        if isinstance(transcript, dict):
            detailed_transcript_path = output_dir / f"{base_name}_transcript_detailed.json"
            with open(detailed_transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)

        # VTT subtitles
        try:
            if isinstance(transcript, dict) and transcript.get("segments"):
                vtt_path = output_dir / f"{base_name}_transcript.vtt"
                with open(vtt_path, "w", encoding="utf-8") as f:
                    f.write("WEBVTT\n\n")
                    for i, segment in enumerate(transcript["segments"]):
                        start_time_str = self._format_timestamp_vtt(segment["start"])
                        end_time_str = self._format_timestamp_vtt(segment["end"])
                        text = segment["text"].strip()
                        f.write(f"{i+1}\n")
                        f.write(f"{start_time_str} --> {end_time_str}\n")
                        f.write(f"{text}\n\n")
        except Exception as e:
            print(f"Warning: Failed to save VTT transcript: {e}")

        # Timing report
        total_time = time.time() - start_time
        timing_info["total_time"] = total_time
        timing_info["video_file"] = str(video_path)
        timing_info["video_size_mb"] = video_path.stat().st_size / (1024 * 1024)

        timing_report_path = output_dir / f"{base_name}_timing_report.json"
        with open(timing_report_path, "w", encoding="utf-8") as f:
            json.dump(timing_info, f, indent=2, ensure_ascii=False)

        save_time = time.time() - save_start
        timing_info["file_saving"] = save_time

        report = self.generate_report(
            output_dir,
            transcript_path,
            timing_info,
        )

        # Cleanup: keep original video, delete extracted audio + chunks
        print(f"Preserving original video file: {video_path}")

        try:
            if audio_path.exists():
                audio_path.unlink()
                print(f"Deleted audio file: {audio_path}")
        except Exception as e:
            print(f"Warning: Failed to delete audio file {audio_path}: {e}")

        if len(audio_chunks) > 1:
            chunk_dir = audio_chunks[0].parent
            for chunk in audio_chunks:
                try:
                    if chunk.exists():
                        chunk.unlink()
                        print(f"Deleted audio chunk: {chunk}")
                except Exception as e:
                    print(f"Warning: Failed to delete audio chunk {chunk}: {e}")

            try:
                if chunk_dir.exists() and not any(chunk_dir.iterdir()):
                    chunk_dir.rmdir()
                    print(f"Deleted chunk directory: {chunk_dir}")
            except Exception as e:
                print(f"Warning: Failed to delete chunk directory {chunk_dir}: {e}")

        return report

    def generate_report(
        self,
        output_path: Path,
        transcript_path: Path,
        timing_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "timing_info": timing_info,
            "status": "completed",
            "output_path": str(output_path),
            "transcript_path": str(transcript_path),
        }

    def _safe_folder_name(self, name: str) -> str:
        invalid_chars = '<>:"/\\|?*'
        for ch in invalid_chars:
            name = name.replace(ch, "_")
        name = " ".join(name.split()).strip()
        return name[:200]


    def _format_timestamp(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_part = int(seconds % 60)
        milliseconds = int(round((seconds - int(seconds)) * 1000))

        if milliseconds == 1000:
            seconds_part += 1
            milliseconds = 0
        if seconds_part == 60:
            minutes += 1
            seconds_part = 0
        if minutes == 60:
            hours += 1
            minutes = 0

        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"

    def _format_timestamp_vtt(self, seconds: float) -> str:
        return self._format_timestamp(seconds)

    def _build_sentences_from_words(self, words: List[dict]) -> List[dict]:
        sentences = []
        current_words = []
        sent_start = None

        for w in words:
            word_text = str(w.get("word", "")).strip()
            if not word_text:
                continue

            if sent_start is None:
                sent_start = float(w["start"])

            current_words.append(w)

            if word_text.endswith((".", "!", "?")):
                sent_end = float(w["end"])
                sentence_text = self._join_words_naturally([x["word"] for x in current_words])

                sentences.append(
                    {
                        "start": sent_start,
                        "end": sent_end,
                        "text": sentence_text.strip(),
                    }
                )

                current_words = []
                sent_start = None

        if current_words:
            sentence_text = self._join_words_naturally([x["word"] for x in current_words])
            sentences.append(
                {
                    "start": sent_start if sent_start is not None else 0.0,
                    "end": float(current_words[-1]["end"]),
                    "text": sentence_text.strip(),
                }
            )

        return sentences

    def _join_words_naturally(self, words: List[str]) -> str:
        text = ""
        no_space_before = {".", ",", "!", "?", ";", ":", "%", ")", "]", "}"}
        no_space_after = {"(", "[", "{"}

        for token in words:
            token = str(token)

            if not text:
                text = token
            elif token in no_space_before:
                text += token
            elif text[-1] in no_space_after:
                text += token
            else:
                text += " " + token

        return text


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.run_pipeline <video_path>")
        sys.exit(1)

    config = Config.from_env()
    logging.basicConfig(level=logging.INFO)

    pipeline = SurgeryTranscriptionPipeline(config)
    video_path = Path(sys.argv[1])

    print(f"\n{'='*60}")
    print(f"Processing video: {video_path.name}")
    print(f"Video size: {video_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"Using whisper backend: {config.whisper_backend}")
    if config.whisper_backend == "local":
        print(f"Local whisper model: {config.whisper_model_name}")
    print(f"{'='*60}\n")

    start_time = time.time()
    results = pipeline.process_video(video_path)
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Results saved to: {results['output_path']}")
    print(f"Transcript saved to: {results['transcript_path']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()