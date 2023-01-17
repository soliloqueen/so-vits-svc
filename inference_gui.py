import io
import os
import logging
import time
import sys
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow,
                               QFrame, QFileDialog, QLineEdit,
                               QPushButton, QVBoxLayout, QLabel)

import librosa
import numpy as np
import soundfile

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")

infer_tool.mkdir(["raw", "results"])
slice_db = -40  
wav_format = 'flac'

class MainWindow (QMainWindow):
    def __init__(self):
        super().__init__()

        self.model_path = "models/placeholder.pth"
        self.config_path = "configs/placeholder.json"
        self.clean_files = [0]

        self.svc_model = []

        self.setWindowTitle("so-vits-svc GUI")
        self.central_widget = QFrame()
        self.layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)
        
        self.header = QLabel("sof-vits")
        self.layout.addWidget(self.header)

        self.model_button = QPushButton("Model")
        self.layout.addWidget(self.model_button)
        self.model_label = QLabel("Current Model: "+
            self.model_path)
        self.layout.addWidget(self.model_label)
        self.model_button.clicked.connect(self.model_dialog)

        self.config_button = QPushButton("Config")
        self.layout.addWidget(self.config_button)
        self.config_label = QLabel("Current Config: "+
            self.config_path)
        self.layout.addWidget(self.config_label)
        self.config_button.clicked.connect(self.config_dialog)

        self.file_button = QPushButton("Files to Convert")
        self.layout.addWidget(self.file_button)
        self.file_label = QLabel("Files: "+str(self.clean_files))
        self.layout.addWidget(self.file_label)
        self.file_button.clicked.connect(self.file_dialog)

        self.transpose_label = QLabel("Transpose")
        self.layout.addWidget(self.transpose_label)
        self.transpose_num = QLineEdit('0')
        self.layout.addWidget(self.transpose_num)
        self.transpose_num.setInputMask('99')

        self.convert_button = QPushButton("Convert")
        self.layout.addWidget(self.convert_button)
        self.convert_button.clicked.connect(self.convert)

    def try_load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.config_path):
            self.svc_model = Svc(self.model_path, self.config_path)
            print ("Loaded model successfully from ",self.model_path)

    def model_file_name(self):
        if self.model_path is None:
            return None
        sp = self.model_path.replace('\\','/').split('/')[-1]
        return str(Path(sp).with_suffix(''))

    def model_dialog(self):
        self.model_path = QFileDialog.getOpenFileName(
            self, "Model", self.model_path)[0]
        self.model_label.setText("Current Model: "+self.model_path)
        self.try_load_model()
        pass

    def config_dialog(self):
        self.config_path = QFileDialog.getOpenFileName(
            self, "Config", self.config_path)[0]
        self.config_label.setText("Current Config: "+self.config_path)
        self.try_load_model()
        pass

    def file_dialog(self):
        self.clean_files = QFileDialog.getOpenFileNames(
            self, "Files to process")[0]
        self.file_label.setText("Files: "+str(self.clean_files))

        # int(self.transpose_num.text())

    def convert(self):
        try:
            trans = int(self.transpose_num.text())
            #print(self.clean_files[0])
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
                        out_audio, out_sr = self.svc_model.infer(0, trans, raw_path)
                        _audio = out_audio.cpu().numpy()
                    audio.extend(list(_audio))

                res_path = f'./results/{wav_name}_{trans}key_{self.model_file_name()}.{wav_format}'
                soundfile.write(res_path, audio, self.svc_model.target_sample, format=wav_format)
        except Exception as e:
            print (e)


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
