import math
import traceback
import io
import os
import logging
import time
import sys
import copy
import importlib.util
from ctypes import cast, POINTER, c_int, c_short, c_float
from pathlib import Path
from PyQt5.QtCore import pyqtSignal, Qt, QUrl
from PyQt5.QtGui import (QIntValidator, QDoubleValidator)
from PyQt5.QtMultimedia import (
   QMediaContent, QMediaPlayer, QAudioRecorder,
   QAudioEncoderSettings, QMultimedia, QAudioDeviceInfo,
   QAudioProbe, QAudioFormat)
from PyQt5.QtWidgets import (QWidget,
   QSizePolicy, QStyle, QProgressBar,
   QApplication, QMainWindow,
   QFrame, QFileDialog, QLineEdit, QSlider,
   QPushButton, QHBoxLayout, QVBoxLayout, QLabel,
   QPlainTextEdit, QComboBox, QGroupBox, QCheckBox)
import numpy as np
import soundfile
import glob
import json
import torch
import subprocess
from datetime import datetime
from collections import deque
from pathlib import Path

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

import librosa

if importlib.util.find_spec("requests"):
    import requests
    REQUESTS_AVAILABLE = True
else:
    REQUESTS_AVAILABLE = False

if (subprocess.run(["where","rubberband"] if os.name == "nt" else 
    ["which","rubberband"]).returncode == 0) and importlib.util.find_spec("pyrubberband"):
    print("Rubberband is available!")
    import pyrubberband as pyrb
    RUBBERBAND_AVAILABLE = True
else:
    print("Rubberband is not available. Timestretch not available.")
    RUBBERBAND_AVAILABLE = False

TALKNET_ADDR = "127.0.0.1:8050"
MODELS_DIR = "models"
RECORD_DIR = "./recordings"
JSON_NAME = "inference_gui2_persist.json"
def get_speakers():
    speakers = []
    for _,dirs,_ in os.walk(MODELS_DIR):
        for folder in dirs:
            cur_speaker = {}
            # Look for G_****.pth
            g = glob.glob(os.path.join(MODELS_DIR,folder,'G_*.pth'))
            if not len(g):
                print("Skipping "+folder+", no G_*.pth")
                continue
            cur_speaker["model_path"] = g[0]
            # Look for config.json
            cfg = glob.glob(os.path.join(MODELS_DIR,folder,'*.json'))
            if not len(cfg):
                print("Skipping "+folder+", no config json")
                continue
            cur_speaker["cfg_path"] = cfg[0]
            with open(cur_speaker["cfg_path"]) as f:
                try:
                    cfg_json = json.loads(f.read())
                except Exception as e:
                    print("Malformed config json in "+folder)
                for name, i in cfg_json["spk"].items():
                    cur_speaker["name"] = name
                    cur_speaker["id"] = i
                    speakers.append(copy.copy(cur_speaker))
    return sorted(speakers, key=lambda x:x["name"].lower())

def el_trunc(s, n=80):
    return s[:min(len(s),n-3)]+'...'

def backtruncate_path(path, n=80):
    if len(path) < (n):
        return path
    path = path.replace('\\','/')
    spl = path.split('/')
    pth = spl[-1]
    i = -1

    while len(pth) < (n - 3):
        i -= 1
        if abs(i) > len(spl):
            break
        pth = os.path.join(spl[i],pth)

    spl = pth.split(os.path.sep)
    pth = os.path.join(*spl)
    return '...'+pth

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")

infer_tool.mkdir(["raw", "results"])
slice_db = -40  
wav_format = 'wav'

class AudioPreviewWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.vlayout = QVBoxLayout(self)

        self.playing_label = QLabel("Preview")
        self.vlayout.addWidget(self.playing_label)

        self.player_frame = QFrame()
        self.vlayout.addWidget(self.player_frame)

        self.player_layout = QHBoxLayout(self.player_frame)

        #self.playing_label.hide()

        self.player = QMediaPlayer()
        self.player.setNotifyInterval(500)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setSizePolicy(QSizePolicy.Expanding,
            QSizePolicy.Preferred)
        self.player_layout.addWidget(self.seek_slider)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(
            getattr(QStyle, 'SP_MediaPlay')))
        self.player_layout.addWidget(self.play_button)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setSizePolicy(QSizePolicy.Minimum,
            QSizePolicy.Minimum)

        self.seek_slider.sliderMoved.connect(self.seek)
        self.player.positionChanged.connect(self.update_seek_slider)
        self.player.stateChanged.connect(self.state_changed)
        self.player.durationChanged.connect(self.duration_changed)

    def set_text(self, text=""):
        if len(text) > 0:
            self.playing_label.show()
            self.playing_label.setText(text)
        else:
            self.playing_label.hide()

    def from_file(self, path):
        self.player.stop()
        if hasattr(self, 'audio_buffer'):
            self.audio_buffer.close()

        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))

        self.play_button.setIcon(self.style().standardIcon(
            getattr(QStyle, 'SP_MediaPlay')))

    def from_memory(self, data):
        self.player.stop()
        if hasattr(self, 'audio_buffer'):
            self.audio_buffer.close()

        self.audio_data = QByteArray(data)
        self.audio_buffer = QBuffer()
        self.audio_buffer.setData(self.audio_data)
        self.audio_buffer.open(QBuffer.ReadOnly)
        player.setMedia(QMediaContent(), self.audio_buffer)

    def state_changed(self, state):
        if (state == QMediaPlayer.StoppedState) or (
            state == QMediaPlayer.PausedState):
            self.play_button.setIcon(self.style().standardIcon(
                getattr(QStyle, 'SP_MediaPlay')))

    def duration_changed(self, dur):
        self.seek_slider.setRange(0, self.player.duration())

    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        elif self.player.mediaStatus() != QMediaPlayer.NoMedia:
            self.player.play()
            self.play_button.setIcon(self.style().standardIcon(
                getattr(QStyle, 'SP_MediaPause')))

    def update_seek_slider(self, position):
        self.seek_slider.setValue(position)

    def seek(self, position):
        self.player.setPosition(position)

class AudioRecorder(QGroupBox):
    def __init__(self, par):
        super().__init__()
        self.setTitle("audio recorder")
        self.setStyleSheet("padding:10px")
        self.layout = QVBoxLayout(self)
        self.ui_parent = par

        self.preview = AudioPreviewWidget()
        self.layout.addWidget(self.preview)

        self.recorder = QAudioRecorder()
        self.input_dev_box = QComboBox()
        for inp in self.recorder.audioInputs():
            if self.input_dev_box.findText(inp) == -1:
                self.input_dev_box.addItem(inp)
        self.layout.addWidget(self.input_dev_box)
        self.input_dev_box.currentIndexChanged.connect(self.set_input_dev)
        self.set_input_dev(0) # Always use the first listed output
        # Doing otherwise on Windows would require platform-specific code

        self.record_button = QPushButton("Record")
        self.record_button.clicked.connect(self.toggle_record)
        self.layout.addWidget(self.record_button)

        self.probe = QAudioProbe()
        self.probe.setSource(self.recorder)
        self.probe.audioBufferProbed.connect(self.update_volume)
        self.volume_meter = QProgressBar()
        self.volume_meter.setTextVisible(False)
        self.volume_meter.setRange(0, 100)
        self.volume_meter.setValue(0)
        self.layout.addWidget(self.volume_meter)

        # RECORD_DIR
        self.record_dir = os.path.abspath(RECORD_DIR)
        self.record_dir_button = QPushButton("Change Recording Directory")
        self.layout.addWidget(self.record_dir_button)
        self.record_dir_label = QLabel("Recordings directory: "+str(
            self.record_dir))
        self.record_dir_button.clicked.connect(self.record_dir_dialog)

        self.audio_settings = QAudioEncoderSettings()
        self.audio_settings.setCodec("audio/pcm")
        self.audio_settings.setSampleRate(44100)
        self.audio_settings.setBitRate(16)
        self.audio_settings.setQuality(QMultimedia.HighQuality)
        self.audio_settings.setEncodingMode(
            QMultimedia.ConstantQualityEncoding)

        self.last_output = ""

        self.sovits_button = QPushButton("Push last output to so-vits-svc")
        self.layout.addWidget(self.sovits_button)
        self.sovits_button.clicked.connect(self.push_to_sovits)

        if (par.talknet_available):
            self.talknet_button = QPushButton("Push last output to TalkNet")
            self.layout.addWidget(self.talknet_button)
            self.talknet_button.clicked.connect(self.push_to_talknet)

        self.layout.addStretch()

    def update_volume(self, buf):
        sample_size = buf.format().sampleSize()
        sample_count = buf.sampleCount()
        ptr = buf.constData()
        ptr.setsize(int(sample_size/8)*sample_count)

        samples = np.asarray(np.frombuffer(ptr, np.int16)).astype(float)
        rms = np.sqrt(np.mean(samples**2))
            
        level = rms / (2 ** 14)

        self.volume_meter.setValue(int(level * 100))

    def set_input_dev(self, idx):
        self.recorder.setAudioInput(self.recorder.audioInputs()[idx])

    def record_dir_dialog(self):
        temp_record_dir = QFileDialog.getExistingDirectory(self,
            "Recordings Directory", self.record_dir, QFileDialog.ShowDirsOnly)
        if not os.path.exists(temp_record_dir): 
            return
        self.record_dir = temp_record_dir
        self.record_dir_label.setText(
            "Recordings directory: "+str(self.record_dir))

    def toggle_record(self):
        if self.recorder.status() == QAudioRecorder.RecordingStatus:
            self.recorder.stop()
            self.record_button.setText("Record")
            self.last_output = self.recorder.outputLocation().toLocalFile()
            self.preview.from_file(
                self.recorder.outputLocation().toLocalFile())
            self.preview.set_text("Preview - "+os.path.basename(
                self.recorder.outputLocation().toLocalFile()))
        else:
            self.record()
            self.record_button.setText("Recording to "+str(
                self.recorder.outputLocation().toLocalFile()))

    def record(self):
        unix_time = time.time()
        self.recorder.setEncodingSettings(self.audio_settings)
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir, exist_ok=True)
        output_name = "rec_"+str(int(unix_time))
        self.recorder.setOutputLocation(QUrl.fromLocalFile(os.path.join(
            self.record_dir,output_name)))
        self.recorder.setContainerFormat("audio/x-wav")
        self.recorder.record()

    def push_to_sovits(self):
        if not os.path.exists(self.last_output):
            return
        self.ui_parent.clean_files = [self.last_output]
        self.ui_parent.update_file_label()
        self.ui_parent.update_input_preview()

    def push_to_talknet(self):
        if not os.path.exists(self.last_output):
            return
        self.ui_parent.talknet_file = self.last_output
        self.ui_parent.talknet_file_label.setText(
            "File: "+str(self.ui_parent.talknet_file))
        self.ui_parent.talknet_update_preview()

class FileButton(QPushButton):
    fileDropped = pyqtSignal(list)
    def __init__(self, label = "Files to Convert"):
        super().__init__(label)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            clean_files = []
            for url in event.mimeData().urls():
                if not url.toLocalFile():
                    continue
                clean_files.append(url.toLocalFile())
            self.fileDropped.emit(clean_files)
            event.acceptProposedAction()
        else:
            event.ignore()
        pass

class InferenceGui2 (QMainWindow):
    def __init__(self):
        super().__init__()


        self.clean_files = [0]
        self.speakers = get_speakers()
        self.speaker = {}
        self.output_dir = os.path.abspath("./results/")
        self.cached_file_dir = os.path.abspath(".")
        self.recent_dirs = deque(maxlen=10)

        self.svc_model = []

        self.setWindowTitle("so-vits-svc GUI")
        self.central_widget = QFrame()
        self.layout = QHBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        self.sovits_frame = QGroupBox(self)
        self.sovits_frame.setTitle("so-vits-svc")
        self.sovits_frame.setStyleSheet("padding:10px")
        self.sovits_lay = QVBoxLayout(self.sovits_frame)
        self.layout.addWidget(self.sovits_frame)

        self.load_persist()
        self.talknet_available = self.try_connect_talknet()

        # Cull non-existent paths from recent_dirs
        self.recent_dirs = deque(
            [d for d in self.recent_dirs if os.path.exists(d)])
        
        self.speaker_box = QComboBox()
        for spk in self.speakers:
            self.speaker_box.addItem(spk["name"])
        self.speaker_label = QLabel("Speaker:")
        self.sovits_lay.addWidget(self.speaker_label)
        self.sovits_lay.addWidget(self.speaker_box)
        self.speaker_box.currentIndexChanged.connect(self.try_load_speaker)
        self.try_load_speaker(0)

        self.file_button = FileButton()
        self.sovits_lay.addWidget(self.file_button)
        self.file_label = QLabel("Files: "+str(self.clean_files))
        self.file_label.setWordWrap(True)
        self.sovits_lay.addWidget(self.file_label)
        self.file_button.clicked.connect(self.file_dialog)
        self.file_button.fileDropped.connect(self.update_files)

        self.input_preview = AudioPreviewWidget()
        self.sovits_lay.addWidget(self.input_preview)

        self.recent_label = QLabel("Recent Directories:")
        self.sovits_lay.addWidget(self.recent_label)
        self.recent_combo = QComboBox()
        self.sovits_lay.addWidget(self.recent_combo)
        self.recent_combo.activated.connect(self.recent_dir_dialog)

        self.transpose_validator = QIntValidator(-24,24)

        # Source pitchshifting
        self.source_transpose_label = QLabel(
            "Formant Shift (half-steps)")
        self.source_transpose_num = QLineEdit('0')
        self.source_transpose_num.setValidator(self.transpose_validator)
        #if PSOLA_AVAILABLE:
        self.sovits_lay.addWidget(self.source_transpose_label)
        self.sovits_lay.addWidget(self.source_transpose_num)

        self.transpose_label = QLabel("Transpose")
        self.sovits_lay.addWidget(self.transpose_label)
        self.transpose_num = QLineEdit('0')
        self.sovits_lay.addWidget(self.transpose_num)
        self.transpose_num.setValidator(self.transpose_validator)

        self.timestretch_validator = QDoubleValidator(0.5,1.0,1)

        if RUBBERBAND_AVAILABLE:
            self.ts_label = QLabel("Timestretch (0.5, 1.0)")
            self.ts_num = QLineEdit('1.0')
            self.ts_num.setValidator(self.timestretch_validator)
            self.sovits_lay.addWidget(self.ts_label)
            self.sovits_lay.addWidget(self.ts_num)

        self.output_button = QPushButton("Change Output Directory")
        self.sovits_lay.addWidget(self.output_button)
        self.output_label = QLabel("Output directory: "+str(self.output_dir))
        self.sovits_lay.addWidget(self.output_label)
        self.output_button.clicked.connect(self.output_dialog)

        self.convert_button = QPushButton("Convert")
        self.sovits_lay.addWidget(self.convert_button)
        self.convert_button.clicked.connect(self.sofvits_convert)

        # TODO right now this only handles the first file processed.
        self.output_preview = AudioPreviewWidget()
        self.sovits_lay.addWidget(self.output_preview)

        self.sovits_lay.addStretch()

        self.audio_recorder = AudioRecorder(self)
        self.layout.addWidget(self.audio_recorder)

        # TalkNet component
        if self.talknet_available:
            self.try_load_talknet()

        self.update_recent_combo()

    def update_files(self, files):
        if (files is None) or (len(files) == 0):
            return
        self.clean_files = files
        self.update_file_label()
        dir_path = os.path.abspath(os.path.dirname(self.clean_files[0]))
        if not dir_path in self.recent_dirs:
            self.recent_dirs.append(dir_path)
        self.update_input_preview()
        self.update_recent_combo()

    # Tests for TalkNet connection and compatibility
    def try_connect_talknet(self):
        import socket
        if not REQUESTS_AVAILABLE:
            print("requests library unavailable; not loading talknet options")
            return False
        spl = self.talknet_addr.split(':')
        if (spl is None) or (len(spl) == 1):
            print("Couldn't parse talknet address "+self.talknet_addr)
            return False
        ip = spl[0]
        port = int(spl[1])
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            result = sock.connect_ex((ip, port))
            if result == 0:
                print("Successfully found TalkNet on address "+self.talknet_addr)
                sock.close()
                return True
            else:
                print("Could not find TalkNet on address "+self.talknet_addr)
                sock.close()
                return False
        except socket.gaierror:
            print("Couldn't connect to talknet address "+self.talknet_addr)
            sock.close()
            return False
        except socket.error:
            print("Couldn't connect to talknet address "+self.talknet_addr)
            sock.close()
            return False
        sock.close()
        return False

    def try_load_talknet(self):
        self.talknet_frame = QGroupBox(self)
        self.talknet_frame.setTitle("talknet")
        self.talknet_frame.setStyleSheet("padding:10px")
        self.talknet_lay = QVBoxLayout(self.talknet_frame)

        self.character_box = QComboBox()
        self.character_label = QLabel("Speaker:")
        response = requests.get('http://'+self.talknet_addr+'/characters')
        if response.status_code == 200:
            try:
                self.talknet_chars = json.loads(response.text)
            except Exception as e:
                print("Couldn't parse TalkNet response.")
                print("Are you running the correct TalkNet server?")
                return

        for k in self.talknet_chars.keys():
            self.character_box.addItem(k)
        self.talknet_lay.addWidget(self.character_label)
        self.talknet_lay.addWidget(self.character_box)
        self.character_box.currentTextChanged.connect(self.talknet_character_load)
        if len(self.talknet_chars.keys()):
            self.cur_talknet_char = next(iter(self.talknet_chars.keys()))
        else:
            self.cur_talknet_char = "N/A"

        self.talknet_file_button = FileButton(label="Provide input audio")
        self.talknet_file = ""
        self.talknet_file_label = QLabel("File: "+self.talknet_file)
        self.talknet_lay.addWidget(self.talknet_file_button)
        self.talknet_lay.addWidget(self.talknet_file_label)
        self.talknet_file_button.clicked.connect(self.talknet_file_dialog)
        self.talknet_file_button.fileDropped.connect(self.talknet_update_file)

        self.talknet_input_preview = AudioPreviewWidget()
        self.talknet_lay.addWidget(self.talknet_input_preview)
        
        self.talknet_recent_label = QLabel("Recent Directories:")
        self.talknet_lay.addWidget(self.talknet_recent_label)
        self.talknet_recent_combo = QComboBox()
        self.talknet_lay.addWidget(self.talknet_recent_combo)
        self.talknet_recent_combo.activated.connect(self.talknet_recent_dir_dialog)

        self.talknet_transfer_sovits = FileButton(
            label='Transfer input to so-vits-svc')
        self.talknet_lay.addWidget(self.talknet_transfer_sovits)
        self.talknet_transfer_sovits.clicked.connect(self.transfer_to_sovits)

        self.talknet_transpose_label = QLabel("Transpose")
        self.talknet_transpose_num = QLineEdit('0')
        self.talknet_transpose_num.setValidator(self.transpose_validator)
        self.talknet_lay.addWidget(self.talknet_transpose_label)
        self.talknet_lay.addWidget(self.talknet_transpose_num)

        self.talknet_transcript_label = QLabel("Transcript")
        self.talknet_transcript_edit = QPlainTextEdit()
        self.talknet_lay.addWidget(self.talknet_transcript_label)
        self.talknet_lay.addWidget(self.talknet_transcript_edit)

        self.talknet_sovits = QCheckBox("Push TalkNet output to so-vits-svc")
        self.talknet_lay.addWidget(self.talknet_sovits)

        self.talknet_gen_button = QPushButton("Generate")
        self.talknet_lay.addWidget(self.talknet_gen_button)
        self.talknet_gen_button.clicked.connect(self.talknet_generate_request)

        self.talknet_output_info = QLabel("--output info (empty)--")
        self.talknet_output_info.setWordWrap(True)
        self.talknet_lay.addWidget(self.talknet_gen_button)
        self.talknet_lay.addWidget(self.talknet_output_info)

        self.talknet_output_preview = AudioPreviewWidget()
        self.talknet_sovits_output_preview = AudioPreviewWidget()
        self.talknet_lay.addWidget(self.talknet_output_preview)
        self.talknet_lay.addWidget(self.talknet_sovits_output_preview)
        self.talknet_sovits_output_preview.hide()

        self.layout.addWidget(self.talknet_frame)
        print("Loaded TalkNet")

        # TODO ? multiple audio preview
        # TODO ? multiple audio selection for TalkNet?

        # TODO optional transcript output?
        # TODO option to disable automatically outputting sound files,
        # or to save in a separate directory.
        # TODO fancy concurrent processing stuff

    def talknet_character_load(self, k):
        self.cur_talknet_char = k

    # Really, it can just drop packets inside the machine?

    def talknet_generate_request(self):
        req_time = datetime.now().strftime("%H:%M:%S")
        response = requests.post('http://'+self.talknet_addr+'/upload',
            data=json.dumps({'char':self.cur_talknet_char,
                'wav':self.talknet_file,
                'transpose':int(self.talknet_transpose_num.text()),
                'transcript':self.talknet_transcript_edit.toPlainText(),
                'results_dir':self.output_dir}),
             headers={'Content-Type':'application/json'})
        if response.status_code != 200:
            print("TalkNet generate request failed.")
            print("It may be useful to check the TalkNet server output.")
            return
        res = json.loads(response.text)
        if self.talknet_sovits.isChecked():
            # Only one file should be outputted here
            sovits_res_path = self.convert([res["output_path"]], dry_trans=0,
                source_trans=0)[0]
            # Assume no transposition is desired since that is handled
            # in TalkNet
        self.talknet_output_preview.from_file(res.get("output_path"))
        self.talknet_output_preview.set_text("Preview - "+res.get(
            "output_path","N/A"))
        if self.talknet_sovits.isChecked():
            self.talknet_output_preview.from_file(sovits_res_path)
            self.talknet_output_preview.set_text("Preview - "+
                sovits_res_path)
        self.talknet_output_info.setText("Last successful request: "+req_time+'\n'+
            "ARPAbet: "+res.get("arpabet","N/A")+'\n'+
            "Output path: "+res.get("output_path","N/A")+'\n')

    def update_file_label(self):
        self.file_label.setText("Files: "+str(self.clean_files))

    def update_input_preview(self):
        self.input_preview.from_file(self.clean_files[0])
        self.input_preview.set_text("Preview - "+self.clean_files[0])

    def transfer_to_sovits(self):
        if (self.talknet_file is None) or not (
            os.path.exists(self.talknet_file)):
            return
        self.clean_files = [self.talknet_file]
        self.update_file_label()

    def try_load_speaker(self, index):
        self.speaker = self.speakers[index]
        print ("Loading "+self.speakers[index]["name"])
        self.svc_model = Svc(self.speakers[index]["model_path"],
            self.speakers[index]["cfg_path"])

    def talknet_file_dialog(self):
        self.talknet_update_file(
            QFileDialog.getOpenFileName(self, "File to process")[0])

    def talknet_update_preview(self):
        self.talknet_input_preview.from_file(self.talknet_file)
        self.talknet_input_preview.set_text("Preview - "+self.talknet_file)

    def talknet_update_file(self, files):
        if (files is None) or (len(files) == 0):
            return
        self.talknet_file = files[0]
        self.talknet_update_preview()
        self.talknet_file_label.setText("File: "+str(self.talknet_file))
        dir_path = os.path.abspath(os.path.dirname(self.talknet_file))
        if not dir_path in self.recent_dirs:
            self.recent_dirs.append(dir_path)
        self.update_recent_combo()

    def file_dialog(self):
        # print("opening file dialog")
        if not self.recent_dirs:
            self.update_files(QFileDialog.getOpenFileNames(
                self, "Files to process")[0])
        else:
            self.update_files(QFileDialog.getOpenFileNames(
                self, "Files to process", self.recent_dirs[-1])[0])

    def recent_dir_dialog(self, index):
        # print("opening dir dialog")
        if not os.path.exists(self.recent_dirs[index]):
            print("Path did not exist: ", self.recent_dirs[index])
        self.update_files(QFileDialog.getOpenFileNames(
            self, "Files to process", self.recent_dirs[index])[0])

    def talknet_recent_dir_dialog(self, index):
        if not os.path.exists(self.recent_dirs[index]):
            print("Path did not exist: ", self.recent_dirs[index])
        self.talknet_update_file(QFileDialog.getOpenFileNames(
            self, "Files to process", self.recent_dirs[index])[0])

    def update_recent_combo(self):
        self.recent_combo.clear()
        if self.talknet_available:
            self.talknet_recent_combo.clear()
        for d in self.recent_dirs:
            self.recent_combo.addItem(backtruncate_path(d))
            if self.talknet_available:
                self.talknet_recent_combo.addItem(backtruncate_path(d))

    def output_dialog(self):
        temp_output_dir = QFileDialog.getExistingDirectory(self,
            "Output Directory", self.output_dir, QFileDialog.ShowDirsOnly)
        if not os.path.exists(output_dir):
            return
        self.output_dir = temp_output_dir
        self.output_label.setText("Output Directory: "+str(self.output_dir))

        # int(self.transpose_num.text())

    def save_persist(self):
        with open(JSON_NAME, "w") as f:
            o = {"recent_dirs": list(self.recent_dirs),
                 "output_dir": self.output_dir}
            json.dump(o,f)

    def load_persist(self):
        if not os.path.exists(JSON_NAME):
            self.recent_dirs = []
            self.output_dirs = "./results/"
            self.talknet_addr = TALKNET_ADDR
            return
        with open(JSON_NAME, "r") as f:
            o = json.load(f)
            self.recent_dirs = deque(o.get("recent_dirs",[]))
            self.output_dir = o.get("output_dir",os.path.abspath("./results/"))
            self.talknet_addr = o.get("talknet_addr",TALKNET_ADDR)

    def push_pitch(self):
        pass
        # TODO

    def sofvits_convert(self):
        res_paths = self.convert(self.clean_files)
        if len(res_paths) > 0:
            self.output_preview.from_file(res_paths[0])
            self.output_preview.set_text("Preview - "+res_paths[0])
        return res_paths

    def convert(self, clean_files = [],
        dry_trans = None,
        source_trans = None):
        res_paths = []
        if dry_trans is None:
            dry_trans = int(self.transpose_num.text())
        if source_trans is None:
            source_trans = int(self.source_transpose_num.text())
        try:
            trans = dry_trans - source_trans
            for clean_name in clean_files:
                clean_name = str(clean_name)
                print(clean_name)
                infer_tool.format_wav(clean_name)
                wav_path = str(Path(clean_name).with_suffix('.wav'))
                wav_name = Path(clean_name).stem
                chunks = slicer.cut(wav_path, db_thresh=slice_db)
                audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)


                audio = []
                for (slice_tag, data) in audio_data:
                    print(f'#=====segment start, '
                        f'{round(len(data)/audio_sr, 3)}s======')
                    if not (source_trans == 0):
                        print ('performing source transpose...')
                        if not RUBBERBAND_AVAILABLE:
                            data = librosa.effects.pitch_shift(
                                data, sr=audio_sr, n_steps=float(source_trans))
                        else:
                            data = pyrb.pitch_shift(
                                data, sr=audio_sr, n_steps=float(source_trans))
                        print ('finished source transpose.')

                    if RUBBERBAND_AVAILABLE and (float(self.ts_num.text()) != 1.0):
                        data = pyrb.time_stretch(data, audio_sr, float(self.ts_num.text()))

                    raw_path = io.BytesIO()
                    soundfile.write(raw_path, data, audio_sr, format="wav")
                    raw_path.seek(0)

                    length = int(np.ceil(len(data) / audio_sr *
                        self.svc_model.target_sample))
                    if slice_tag:
                        print('jump empty segment')
                        _audio = np.zeros(length)
                    else:
                        out_audio, out_sr = self.svc_model.infer(
                            self.speaker["id"], trans, raw_path)
                        _audio = out_audio.cpu().numpy()
                    audio.extend(list(_audio))

                res_path = os.path.join(self.output_dir,
                    f'{wav_name}_{source_trans}_{dry_trans}key_'
                    f'{self.speaker["name"]}.{wav_format}')

                # Could be made more efficient
                i = 1
                while os.path.exists(res_path):
                    res_path = os.path.join(self.output_dir,
                        f'{wav_name}_{source_trans}_{dry_trans}key_'
                        f'{self.speaker["name"]}{i}.{wav_format}')
                    i += 1

                # if RUBBERBAND_AVAILABLE and (float(self.ts_num.text()) != 1.0):
                    # audio = pyrb.time_stretch(np.array(audio),
                        # audio_sr, 1.0/float(self.ts_num.text()))
                    
                soundfile.write(res_path, audio, self.svc_model.target_sample,
                    format=wav_format)
                res_paths.append(res_path)
        except Exception as e:
            traceback.print_exc()
        return res_paths


app = QApplication(sys.argv)
w = InferenceGui2()
w.show()
app.exec()
w.save_persist()
