# a wrapping of https://github.com/openai/whisper
from enum import Enum
from pathlib import Path
from datetime import timedelta

import whisper
from pydub import AudioSegment

class Timing:
    _hours      :int = None
    _minutes    :int = None
    _seconds    :int = None
    _mseconds   :int = None
    _origin     :float = None

    def __init__(self, seconds: float):
        self._origin = seconds
        delta = timedelta(seconds=seconds)
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds_int = divmod(remainder, 60)

        self._hours = hours
        self._minutes = minutes
        self._seconds = seconds_int
        self._mseconds = int((seconds - total_seconds) * 1000)


    def to_string(self) -> str:
        return f"{self._hours:02}:{self._minutes:02}:{self._seconds:02},{self._mseconds:03}"

    def __str__(self):
        return self.to_string()

    def __eq__(self, time):
        if not isinstance(time, Timing): return False
        return (self._origin - time._origin) < 0.0001

    def __lt__(self, time):
        if not isinstance(time, Timing): return None
        return (self._origin - time._origin) < 0

    def __gt__(self, time):
        if not isinstance(time, Timing): return None
        return (self._origin - time._origin) > 0

    def hours(self)-> int:
        return self._hours

    def minutes(self) -> int:
        return self._minutes

    def seconds(self) -> int:
        return self._seconds

    def milliseconds(self) -> int:
        return self._mseconds



class VttModelType(Enum):
    TINY = 0
    BASE = 1
    SMALL = 2
    MEDIUM = 3
    LARGE = 4
    TURBO = 5

    @staticmethod
    def names():
        return "tiny", "base", "small", "medium", "large", "turbo"

    def __init__(self, value):
        if value >= len(self.names()) or self.value < 0:
            raise ValueError("Invalid model size")
        self.val = value

    @staticmethod
    def from_string(s: str):
        try:
            idx = VttModelType.names().index(s)
        except ValueError:
            raise ValueError("Invalid model size")

        return VttModelType(idx)

    def to_string(self):
        return self.names()[self.val]

    def __str__(self):
        return self.to_string()

class VTT:
    def __init__(self, model_size: VttModelType, init_model: bool = True):
        self.model_size = model_size
        if init_model: self.model_init()

    def model_init(self, in_memory: bool = True):
        self.model = whisper.load_model(self.model_size.to_string(), in_memory=in_memory)

    def transcribe(self, audio_path: Path, lang: str = "zh", verbose=True) -> map:
        verbose = False if verbose is True else None
        result = self.model.transcribe(str(audio_path), verbose=verbose, language=lang)
        return map(lambda r: {"start": r["start"], "end": r["end"], "text": r["text"]}, result["segments"])

    @staticmethod
    def get_audio_time(audio_path: Path) -> Timing:
        audio = AudioSegment.from_file(audio_path, format=audio_path.suffix[1:])
        return Timing(float(len(audio)) / 1000)
    @staticmethod
    def write_srt(segments: map, audio_len: Timing, output_path: Path):
        with output_path.open("w", encoding="utf-8") as f:
            for i, segment in enumerate(segments):
                start = Timing(segment["start"])
                end = Timing(segment["end"])
                if end > audio_len: break
                f.write(f"{i+1}\n{start.to_string()} --> {end.to_string()}\n{segment['text']}\n\n")

    def transcribe_to_srt(self, audio: Path, srt: Path, lang: str = "Mandarin", verbose=True):
        if not audio.exists(): raise ValueError("Audio file does not exist")
        if not srt.exists(): srt.touch()
        # print("[INFO] Parsing audio...")
        audio_len = VTT.get_audio_time(audio)
        # print("[INFO] Start transcribing...")
        segments = self.transcribe(audio, lang, verbose)
        # print("[INFO] Transcribe done, Writing SRT file...")
        VTT.write_srt(segments, audio_len, srt)
        # print("[INFO] SRT file written!")

def read_srt(srt: Path) -> list[str]:
    with open(srt, "r", encoding="utf-8") as f:
        res = []
        line_code = 1
        state = 0       # 0: line_code, 1: time_duration, 2: text
        while True:
            line = f.readline()
            if not line: break
            line = line.rstrip()
            match state:
                case 0:
                    if line.isdigit():
                        if line_code != int(line): continue
                        line_code += 1
                        state = 1
                case 1:
                    if state == 1:
                        state = 2
                case 2:
                    if state == 2:
                        res.append(line)
                        state = 0
                case x:
                    raise ValueError(f"Invalid state {x}")
        return res
