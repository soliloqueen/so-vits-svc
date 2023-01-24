import math
import traceback
import io
import os
import logging
import time
import sys
import copy
import importlib.util
from pathlib import Path
import PyQt5.QtCore as QtCore
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (QApplication, QMainWindow,
                               QFrame, QFileDialog, QLineEdit,
                               QPushButton, QVBoxLayout, QLabel,
                               QComboBox)
import numpy as np
import soundfile
import glob
import json
import torch
from collections import deque
from pathlib import Path

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

#PSOLA_AVAILABLE = False
#if importlib.util.find_spec('psola'):
    #PSOLA_AVAILABLE = True
    #import psola
import librosa

MODELS_DIR = "models"
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
            cfg = glob.glob(os.path.join(MODELS_DIR,folder,'config.json'))
            if not len(cfg):
                print("Skipping "+folder+", no config.json")
                continue
            cur_speaker["cfg_path"] = cfg[0]
            with open(cur_speaker["cfg_path"]) as f:
                try:
                    cfg_json = json.loads(f.read())
                except Exception as e:
                    print("Malformed config.json in "+folder)
                for name, i in cfg_json["spk"].items():
                    cur_speaker["name"] = name
                    cur_speaker["id"] = i
                    speakers.append(copy.copy(cur_speaker))
    return sorted(speakers, key=lambda x:x["name"].lower())

def backtruncate_path(path, n=60):
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
wav_format = 'flac'

class FileButton(QPushButton):
    fileDropped = QtCore.pyqtSignal(list)
    def __init__(self):
        super().__init__("Files to Convert")
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
        self.load_persist()

        # Cull non-existent paths from recent_dirs
        self.recent_dirs = deque(
            [d for d in self.recent_dirs if os.path.exists(d)])

        self.svc_model = []

        self.setWindowTitle("so-vits-svc GUI")
        self.central_widget = QFrame()
        self.layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)
        
        self.header = QLabel("sof-vits")
        self.layout.addWidget(self.header)

        self.speaker_box = QComboBox()
        for spk in self.speakers:
            self.speaker_box.addItem(spk["name"])
        self.speaker_label = QLabel("Speaker:")
        self.layout.addWidget(self.speaker_label)
        self.layout.addWidget(self.speaker_box)
        self.speaker_box.currentIndexChanged.connect(self.try_load_speaker)
        self.try_load_speaker(0)

        self.file_button = FileButton()
        self.layout.addWidget(self.file_button)
        self.file_label = QLabel("Files: "+str(self.clean_files))
        self.layout.addWidget(self.file_label)
        self.file_button.clicked.connect(self.file_dialog)
        self.file_button.fileDropped.connect(self.update_files)

        self.recent_label = QLabel("Recent Directories:")
        self.layout.addWidget(self.recent_label)
        self.recent_combo = QComboBox()
        self.layout.addWidget(self.recent_combo)
        self.update_recent_combo()
        self.recent_combo.activated.connect(self.recent_dir_dialog)

        self.transpose_validator = QIntValidator(-24,24)

        # Source pitchshifting
        self.source_transpose_label = QLabel(
            "Formant Shift (half-steps) (low-quality)")
        self.source_transpose_num = QLineEdit('0')
        self.source_transpose_num.setValidator(self.transpose_validator)
        #if PSOLA_AVAILABLE:
        self.layout.addWidget(self.source_transpose_label)
        self.layout.addWidget(self.source_transpose_num)

        self.transpose_label = QLabel("Transpose")
        self.layout.addWidget(self.transpose_label)
        self.transpose_num = QLineEdit('0')
        self.layout.addWidget(self.transpose_num)
        self.transpose_num.setValidator(self.transpose_validator)

        self.output_button = QPushButton("Output Directory")
        self.layout.addWidget(self.output_button)
        self.output_label = QLabel("Output directory: "+str(self.output_dir))
        self.layout.addWidget(self.output_label)
        self.output_button.clicked.connect(self.output_dialog)

        self.convert_button = QPushButton("Convert")
        self.layout.addWidget(self.convert_button)
        self.convert_button.clicked.connect(self.convert)

    def update_files(self, files):
        if (files is None) or (len(files) == 0):
            return
        self.clean_files = files
        self.file_label.setText("Files: "+str(self.clean_files))
        dir_path = os.path.abspath(os.path.dirname(self.clean_files[0]))
        if not dir_path in self.recent_dirs:
            self.recent_dirs.append(dir_path)
        self.update_recent_combo()

    def try_load_speaker(self, index):
        self.speaker = self.speakers[index]
        print ("Loading "+self.speakers[index]["name"])
        self.svc_model = Svc(self.speakers[index]["model_path"],
            self.speakers[index]["cfg_path"])

    def file_dialog(self):
        print("opening file dialog")
        if not self.recent_dirs:
            self.update_files(QFileDialog.getOpenFileNames(
                self, "Files to process")[0])
        else:
            self.update_files(QFileDialog.getOpenFileNames(
                self, "Files to process", self.recent_dirs[-1])[0])

    def recent_dir_dialog(self, index):
        print("opening dir dialog")
        if not os.path.exists(self.recent_dirs[index]):
            print("Path did not exist: ", self.recent_dirs[index])
        self.update_files(QFileDialog.getOpenFileNames(
            self, "Files to process", self.recent_dirs[index])[0])

    def update_recent_combo(self):
        self.recent_combo.clear()
        for d in self.recent_dirs:
            self.recent_combo.addItem(backtruncate_path(d))

    def output_dialog(self):
        self.output_dir = QFileDialog.getExistingDirectory(self,
            "Output Directory", self.output_dir, QFileDialog.ShowDirsOnly)
        self.output_label.setText("Output Directory: "+str(self.output_dir))

        # int(self.transpose_num.text())

    def save_persist(self):
        with open(JSON_NAME, "w") as f:
            o = {"recent_dirs": list(self.recent_dirs),
                 "output_dir": self.output_dir}
            json.dump(o,f)

    def load_persist(self):
        if not os.path.exists(JSON_NAME):
            return
        with open(JSON_NAME, "r") as f:
            o = json.load(f)
            self.recent_dirs = deque(o["recent_dirs"])
            self.output_dir = o["output_dir"]

    def push_pitch(self):
        pass
        # TODO

    def convert(self):
        try:
            source_trans = int(self.source_transpose_num.text())
            dry_trans = int(self.transpose_num.text())
            trans = int(self.transpose_num.text()) - source_trans
            for clean_name in self.clean_files:
                infer_tool.format_wav(clean_name)
                wav_path = Path(clean_name).with_suffix('.wav')
                wav_name = Path(clean_name).stem
                chunks = slicer.cut(wav_path, db_thresh=slice_db)
                audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

                audio = []
                for (slice_tag, data) in audio_data:

                    print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
                    #if PSOLA_AVAILABLE and not (source_trans == 0):
                    if not (source_trans == 0):
                        print ('performing source transpose...')
                        data = librosa.effects.pitch_shift(data, sr=audio_sr, n_steps=float(source_trans))
                        print ('finished source transpose.')

                    raw_path = io.BytesIO()
                    soundfile.write(raw_path, data, audio_sr, format="wav")
                    raw_path.seek(0)

                    length = int(np.ceil(len(data) / audio_sr * self.svc_model.target_sample))
                    if slice_tag:
                        print('jump empty segment')
                        _audio = np.zeros(length)
                    else:
                        out_audio, out_sr = self.svc_model.infer(self.speaker["id"], trans, raw_path)
                        _audio = out_audio.cpu().numpy()
                    audio.extend(list(_audio))

                #model_base = Path(os.path.basename(self.speaker["model_path"])).with_suffix('')
                res_path = os.path.join(self.output_dir,
                    f'{wav_name}_{source_trans}_{dry_trans}key_{self.speaker["name"]}.{wav_format}')
                soundfile.write(res_path, audio, self.svc_model.target_sample, format=wav_format)
        except Exception as e:
            traceback.print_exc()


app = QApplication(sys.argv)
w = InferenceGui2()
w.show()
app.exec()
w.save_persist()
