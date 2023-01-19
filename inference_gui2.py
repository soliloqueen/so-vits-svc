import traceback
import io
import os
import logging
import time
import sys
import copy
from pathlib import Path
import PySide6.QtCore as QtCore
from PySide6.QtWidgets import (QApplication, QMainWindow,
                               QFrame, QFileDialog, QLineEdit,
                               QPushButton, QVBoxLayout, QLabel,
                               QComboBox)
import numpy as np
import soundfile
import glob
import json
from pathlib import Path

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

MODELS_DIR = "models"
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
logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")

infer_tool.mkdir(["raw", "results"])
slice_db = -40  
wav_format = 'flac'

class FileButton(QPushButton):
    fileDropped = QtCore.Signal(list)
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

class MainWindow (QMainWindow):

    def __init__(self):
        super().__init__()

        self.clean_files = [0]
        self.speakers = get_speakers()
        self.speaker = {}

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

        self.transpose_label = QLabel("Transpose")
        self.layout.addWidget(self.transpose_label)
        self.transpose_num = QLineEdit('0')
        self.layout.addWidget(self.transpose_num)
        self.transpose_num.setInputMask('99')

        self.convert_button = QPushButton("Convert")
        self.layout.addWidget(self.convert_button)
        self.convert_button.clicked.connect(self.convert)

    def update_files(self, files):
        self.clean_files = files
        self.file_label.setText("Files: "+str(self.clean_files))

    def try_load_speaker(self, index):
        self.speaker = self.speakers[index]
        print ("Loading "+self.speakers[index]["name"])
        self.svc_model = Svc(self.speakers[index]["model_path"],
            self.speakers[index]["cfg_path"])

    def file_dialog(self):
        self.clean_files = QFileDialog.getOpenFileNames(
            self, "Files to process")[0]
        self.file_label.setText("Files: "+str(self.clean_files))

        # int(self.transpose_num.text())

    def convert(self):
        try:
            trans = int(self.transpose_num.text())
            for clean_name in self.clean_files:
                infer_tool.format_wav(clean_name)
                wav_path = Path(clean_name).with_suffix('.wav')
                wav_name = Path(clean_name).stem
                chunks = slicer.cut(wav_path, db_thresh=slice_db)
                audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

                audio = []
                for (slice_tag, data) in audio_data:
                    print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
                    length = int(np.ceil(len(data) / audio_sr * self.svc_model.target_sample))
                    raw_path = io.BytesIO()
                    soundfile.write(raw_path, data, audio_sr, format="wav")
                    raw_path.seek(0)
                    if slice_tag:
                        print('jump empty segment')
                        _audio = np.zeros(length)
                    else:
                        out_audio, out_sr = self.svc_model.infer(self.speaker["id"], trans, raw_path)
                        _audio = out_audio.cpu().numpy()
                    audio.extend(list(_audio))

                #model_base = Path(os.path.basename(self.speaker["model_path"])).with_suffix('')
                res_path = f'./results/{wav_name}_{trans}key_{self.speaker["name"]}.{wav_format}'
                soundfile.write(res_path, audio, self.svc_model.target_sample, format=wav_format)
        except Exception as e:
            traceback.print_exc()


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
