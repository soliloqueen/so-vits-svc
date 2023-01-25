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
from PyQt5.QtGui import (QIntValidator)
from PyQt5.QtWidgets import (QApplication, QMainWindow,
   QFrame, QFileDialog, QLineEdit,
   QPushButton, QHBoxLayout, QVBoxLayout, QLabel,
   QPlainTextEdit, QComboBox, QGroupBox, QCheckBox)
import numpy as np
import soundfile
import glob
import json
import torch
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

TALKNET_ADDR = "127.0.0.1:8050"
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
        self.sovits_lay.addWidget(self.file_label)
        self.file_button.clicked.connect(self.file_dialog)
        self.file_button.fileDropped.connect(self.update_files)

        self.recent_label = QLabel("Recent Directories:")
        self.sovits_lay.addWidget(self.recent_label)
        self.recent_combo = QComboBox()
        self.sovits_lay.addWidget(self.recent_combo)
        self.recent_combo.activated.connect(self.recent_dir_dialog)

        self.transpose_validator = QIntValidator(-24,24)

        # Source pitchshifting
        self.source_transpose_label = QLabel(
            "Formant Shift (half-steps) (low-quality)")
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

        self.output_button = QPushButton("Change Output Directory")
        self.sovits_lay.addWidget(self.output_button)
        self.output_label = QLabel("Output directory: "+str(self.output_dir))
        self.sovits_lay.addWidget(self.output_label)
        self.output_button.clicked.connect(self.output_dialog)

        self.convert_button = QPushButton("Convert")
        self.sovits_lay.addWidget(self.convert_button)
        self.convert_button.clicked.connect(self.sofvits_convert)

        # TalkNet component
        if self.talknet_available:
            self.try_load_talknet()

        self.update_recent_combo()

    def update_files(self, files):
        if (files is None) or (len(files) == 0):
            return
        self.clean_files = files
        self.file_label.setText("Files: "+str(self.clean_files))
        dir_path = os.path.abspath(os.path.dirname(self.clean_files[0]))
        if not dir_path in self.recent_dirs:
            self.recent_dirs.append(dir_path)
        self.update_recent_combo()

    # Tests for TalkNet connection and compatibility
    def try_connect_talknet(self):
        import socket
        if not REQUESTS_AVAILABLE:
            print("requests library unavailable; not loading talknet options")
            return False
        if not self.talknet_addr:
            self.talknet_addr = TALKNET_ADDR
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
        
        self.talknet_recent_label = QLabel("Recent Directories:")
        self.talknet_lay.addWidget(self.talknet_recent_label)
        self.talknet_recent_combo = QComboBox()
        self.talknet_lay.addWidget(self.talknet_recent_combo)
        self.talknet_recent_combo.activated.connect(self.talknet_recent_dir_dialog)

        self.talknet_transfer_sovits = FileButton(label='Transfer input to so-vits-svc')
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

        self.layout.addWidget(self.talknet_frame)
        print("Loaded TalkNet")

        # TODO previews

        # TODO optional transcript output?
        # TODO should be able to specify output directory
        # TODO fancy concurrent processing stuff

    def talknet_character_load(self, k):
        self.cur_talknet_char = k

    def talknet_generate_request(self):
        req_time = datetime.now().strftime("%H:%M:%S")
        print(self.cur_talknet_char)
        response = requests.post('http://'+self.talknet_addr+'/upload',
            data=json.dumps({'char':self.cur_talknet_char,
                'wav':self.talknet_file,
                'transcript':self.talknet_transcript_edit.toPlainText(),
                'results_dir':self.output_dir}),
             headers={'Content-Type':'application/json'})
        if response.status_code != 200:
            print("TalkNet generate request failed.")
            print("It may be useful to check the TalkNet server output.")
            return
        res = json.loads(response.text)
        if self.talknet_sovits.isChecked():
            res_path = self.convert([res["output_path"]], dry_trans=0,
                source_trans=0) 
            # Assume no transposition is desired since that is handled
            # in TalkNet
        self.talknet_output_info.setText("Last successful request: "+req_time+'\n'+
            "ARPAbet: "+res.get("arpabet","N/A")+'\n'+
            "Output path: "+res.get("output_path","N/A")+'\n')

    def transfer_to_sovits(self):
        if (self.talknet_file is None) or not (os.path.exists(self.talknet_file)):
            return
        self.clean_files = [self.talknet_file]
        self.file_label.setText("Files: "+str(self.clean_files))

    def try_load_speaker(self, index):
        self.speaker = self.speakers[index]
        print ("Loading "+self.speakers[index]["name"])
        self.svc_model = Svc(self.speakers[index]["model_path"],
            self.speakers[index]["cfg_path"])

    def talknet_file_dialog(self):
        self.talknet_update_file(
            QFileDialog.getOpenFileName(self, "File to process")[0])

    def talknet_update_file(self, files):
        if (files is None) or (len(files) == 0):
            return
        self.talknet_file = files[0]
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
            self.recent_dirs = deque(o.get("recent_dirs",[]))
            self.output_dir = o.get("output_dir",os.path.abspath("./results/"))
            self.talknet_addr = o.get("talknet_addr", TALKNET_ADDR)

    def push_pitch(self):
        pass
        # TODO

    def sofvits_convert(self):
        return self.convert(self.clean_files)

    def convert(self, clean_files = [],
        dry_trans = None,
        source_trans = None):
        if not dry_trans:
            dry_trans = int(self.transpose_num.text())
        if not source_trans:
            source_trans = int(self.source_transpose_num.text())
        try:
            trans = dry_trans - source_trans
            for clean_name in clean_files:
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
                return res_path
        except Exception as e:
            traceback.print_exc()


app = QApplication(sys.argv)
w = InferenceGui2()
w.show()
app.exec()
w.save_persist()
