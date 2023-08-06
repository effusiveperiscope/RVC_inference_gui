import os
import copy
import sys
import traceback
import time
import numpy as np
import json
import importlib.util
import torch
import glob
from pathlib import Path
from collections import deque
from fairseq import checkpoint_utils
from scipy.io import wavfile
from PyQt5.QtCore import (pyqtSignal, Qt, QUrl, QSize, QMimeData, QMetaObject,
    pyqtSlot)
from PyQt5.QtGui import (QIntValidator, QDoubleValidator, QKeySequence,
    QDrag)
from PyQt5.QtMultimedia import (
   QMediaContent, QAudio, QAudioDeviceInfo, QMediaPlayer, QAudioRecorder,
   QAudioEncoderSettings, QMultimedia,
   QAudioProbe, QAudioFormat)
from PyQt5.QtWidgets import (QWidget,
   QSizePolicy, QStyle, QProgressBar,
   QApplication, QMainWindow,
   QFrame, QFileDialog, QLineEdit, QSlider,
   QPushButton, QHBoxLayout, QVBoxLayout, QLabel,
   QPlainTextEdit, QComboBox, QGroupBox, QCheckBox, QShortcut, QDialog)


now_dir = os.getcwd()
sys.path.append(now_dir)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)

from my_utils import load_audio
from vc_infer_pipeline import VC
from config import Config
config = Config()
from infer_pack.models import (SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
    SynthesizerTrnMs256NSFsid_nono)

MODELS_DIR = "models"
F0_METHODS = ["harvest","pm"]
RECORD_DIR = "./recordings"
RECORD_SHORTCUT = "ctrl+shift+r"
JSON_NAME = "inference_gui_rvc_persist.json"
RECENT_DIR_MAXLEN = 10
if (importlib.util.find_spec("pygame")):
    from pygame import mixer, _sdl2 as devicer
    import pygame._sdl2.audio as sdl2_audio
    print("Automatic mode enabled. Press "+RECORD_SHORTCUT+
        " to toggle recording.")
    PYGAME_AVAILABLE = True
else:
    print("Note: Automatic mode not available."
    "To enable: pip install pygame keyboard")
    PYGAME_AVAILABLE = False

# VST support disabled for now because I don't feel like it
PEDALBOARD_AVAILABLE = False

# get_weights()
def get_voices():
    voices = []
    for _,dirs,_ in os.walk(MODELS_DIR):
        for folder in dirs:
            cur_speaker = {}

            # The G_ and D_ files should NOT be included in this
            gen = glob.glob(os.path.join(MODELS_DIR,folder,'G_*.pth'))
            disc = glob.glob(os.path.join(MODELS_DIR,folder,'D_*.pth'))

            wt = [x for x in glob.glob(
                os.path.join(MODELS_DIR, folder,'*.pth'))
                if (x not in gen) and (x not in disc)]
            if not len(wt):
                print("Skipping "+folder+", no *.pth (weight file)")
                continue

            cur_speaker["weight_path"] = wt[0]
            cur_speaker["model_folder"] = folder

            feature_index = glob.glob(os.path.join(
                MODELS_DIR,folder,'*.index'))
            if not len(feature_index):
                print("Note: No feature search files found for "+folder)
                continue
            else:
                cur_speaker["feature_index"] = feature_index[0]

            voices.append(copy.copy(cur_speaker))
    return voices

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

class FieldWidget(QFrame):
    def __init__(self, label, field):
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0,0,0,0)
        label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(label)
        field.setAlignment(Qt.AlignRight)
        field.sizeHint = lambda: QSize(60, 32)
        field.setSizePolicy(QSizePolicy.Maximum,
            QSizePolicy.Preferred)
        self.layout.addWidget(field)

class AudioPreviewWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.vlayout = QVBoxLayout(self)
        self.vlayout.setSpacing(0)
        self.vlayout.setContentsMargins(0,0,0,0)

        self.playing_label = QLabel("Preview")
        self.playing_label.setWordWrap(True)
        self.vlayout.addWidget(self.playing_label)

        self.player_frame = QFrame()
        self.vlayout.addWidget(self.player_frame)

        self.player_layout = QHBoxLayout(self.player_frame)
        self.player_layout.setSpacing(4)
        self.player_layout.setContentsMargins(0,0,0,0)

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
        self.play_button.setSizePolicy(QSizePolicy.Maximum,
            QSizePolicy.Minimum)
        self.play_button.mouseMoveEvent = self.drag_hook

        self.seek_slider.sliderMoved.connect(self.seek)
        self.player.positionChanged.connect(self.update_seek_slider)
        self.player.stateChanged.connect(self.state_changed)
        self.player.durationChanged.connect(self.duration_changed)

        self.local_file = ""

    def set_text(self, text=""):
        if len(text) > 0:
            self.playing_label.show()
            self.playing_label.setText(text)
        else:
            self.playing_label.hide()

    def from_file(self, path):
        try:
            self.player.stop()
            if hasattr(self, 'audio_buffer'):
                self.audio_buffer.close()

            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(
                os.path.abspath(path))))

            self.play_button.setIcon(self.style().standardIcon(
                getattr(QStyle, 'SP_MediaPlay')))

            self.local_file = path
        except Exception as e:
            pass

    def drag_hook(self, e):
        if e.buttons() != Qt.LeftButton:
            return
        if not len(self.local_file):
            return

        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(
            os.path.abspath(self.local_file))])
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec_(Qt.CopyAction)

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

class AudioRecorderAndVSTs(QGroupBox):
    keyboardRecordSignal = pyqtSignal()
    def __init__(self, par):
        super().__init__()
        self.setTitle("Audio recorder and VST processing")
        self.setStyleSheet("padding:10px")
        self.layout = QVBoxLayout(self)
        self.ui_parent = par

        self.audio_settings = QAudioEncoderSettings()
        if os.name == "nt":
            self.audio_settings.setCodec("audio/pcm")
        else:
            self.audio_settings.setCodec("audio/x-raw")
        self.audio_settings.setSampleRate(44100)
        self.audio_settings.setBitRate(16)
        self.audio_settings.setQuality(QMultimedia.HighQuality)
        self.audio_settings.setEncodingMode(
            QMultimedia.ConstantQualityEncoding)

        self.preview = AudioPreviewWidget()
        self.layout.addWidget(self.preview)

        self.recorder = QAudioRecorder()
        self.input_dev_box = QComboBox()
        self.input_dev_box.setSizePolicy(QSizePolicy.Preferred,
            QSizePolicy.Preferred)
        if os.name == "nt":
            self.audio_inputs = self.recorder.audioInputs()
        else:
            self.audio_inputs = [x.deviceName() 
                for x in QAudioDeviceInfo.availableDevices(0)]

        self.record_button = QPushButton("Record")
        self.record_button.clicked.connect(self.toggle_record)
        self.layout.addWidget(self.record_button)

        for inp in self.audio_inputs:
            if self.input_dev_box.findText(el_trunc(inp,60)) == -1:
                self.input_dev_box.addItem(el_trunc(inp,60))
        self.layout.addWidget(self.input_dev_box)
        self.input_dev_box.currentIndexChanged.connect(self.set_input_dev)
        if len(self.audio_inputs) == 0:
            self.record_button.setEnabled(False) 
            print("No audio inputs found")
        else:
            self.set_input_dev(0) # Always use the first listed output
        # Doing otherwise on Windows would require platform-specific code

        if PYGAME_AVAILABLE and importlib.util.find_spec("keyboard"):
            try:
                print("Keyboard module loaded.")
                print("Recording shortcut without window focus enabled.")
                import keyboard
                def keyboard_record_hook():
                    self.keyboardRecordSignal.emit()
                keyboard.add_hotkey(RECORD_SHORTCUT,keyboard_record_hook)
                self.keyboardRecordSignal.connect(self.toggle_record)
            except ImportError as e:
                print("Keyboard module failed to import.")
                print("On Linux, must be run as root for recording"
                    "hotkey out of focus.")
                self.record_shortcut = QShortcut(QKeySequence(RECORD_SHORTCUT),
                    self)
                self.record_shortcut.activated.connect(self.toggle_record)
        else:
            print("No keyboard module available.")
            print("Using default input capture for recording shortcut.")
            self.record_shortcut = QShortcut(QKeySequence(RECORD_SHORTCUT),
                self)
            self.record_shortcut.activated.connect(self.toggle_record)

        self.probe = QAudioProbe()
        self.probe.setSource(self.recorder)
        self.probe.audioBufferProbed.connect(self.update_volume)
        self.volume_meter = QProgressBar()
        self.volume_meter.setTextVisible(False)
        self.volume_meter.setRange(0, 100)
        self.volume_meter.setValue(0)
        self.layout.addWidget(self.volume_meter)

        if PYGAME_AVAILABLE:
            self.record_out_label = QLabel("Output device")
            mixer.init()
            self.out_devs = sdl2_audio.get_audio_device_names(False)
            mixer.quit()
            self.output_dev_box = QComboBox()
            self.output_dev_box.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Preferred)
            for dev in self.out_devs:
                if self.output_dev_box.findText(el_trunc(dev,60)) == -1:
                    self.output_dev_box.addItem(el_trunc(dev,60))
            self.output_dev_box.currentIndexChanged.connect(self.set_output_dev)
            self.selected_dev = None
            self.set_output_dev(0)
            self.layout.addWidget(self.record_out_label)
            self.layout.addWidget(self.output_dev_box)

        # RECORD_DIR
        self.record_dir = os.path.abspath(RECORD_DIR)
        self.record_dir_button = QPushButton("Change Recording Directory")
        self.layout.addWidget(self.record_dir_button)
        self.record_dir_label = QLabel("Recordings directory: "+str(
            self.record_dir))
        self.record_dir_button.clicked.connect(self.record_dir_dialog)

        self.last_output = ""

        self.rvc_button = QPushButton("Push last output to rvc")
        self.layout.addWidget(self.rvc_button)
        self.rvc_button.clicked.connect(self.push_to_rvc)

        self.automatic_checkbox = QCheckBox("Send automatically")
        self.layout.addWidget(self.automatic_checkbox)

        if PYGAME_AVAILABLE:
            self.mic_checkbox = QCheckBox("Auto-play output to selected output device")
            self.layout.addWidget(self.mic_checkbox)
            self.mic_checkbox.stateChanged.connect(self.update_init_audio)

            self.mic_output_control = QCheckBox("Auto-delete audio from "
                "recordings/results after auto-playing")
            self.layout.addWidget(self.mic_output_control)
            self.mic_output_control.stateChanged.connect(self.update_delfiles)

        if PEDALBOARD_AVAILABLE:
            self.vst_input_frame = QGroupBox(self)
            self.vst_input_frame.setTitle("so-vits-svc Pre VSTs")
            self.vst_input_layout = QVBoxLayout(self.vst_input_frame)
            self.layout.addWidget(self.vst_input_frame)
            self.vst_inputs = []
            for i in range(2):
                vst_widget = VSTWidget()
                self.vst_inputs.append(vst_widget)
                self.vst_input_layout.addWidget(vst_widget)
                vst_widget.sig_editor_open.connect(
                    self.ui_parent.pass_editor_ctl)

            self.vst_output_frame = QGroupBox(self)
            self.vst_output_frame.setTitle("so-vits-svc Post VSTs")
            self.vst_output_layout = QVBoxLayout(self.vst_output_frame)
            self.layout.addWidget(self.vst_output_frame)
            self.vst_outputs = []
            for i in range(2):
                vst_widget = VSTWidget()
                self.vst_outputs.append(vst_widget)
                self.vst_output_layout.addWidget(vst_widget)
                vst_widget.sig_editor_open.connect(
                    self.ui_parent.pass_editor_ctl)
        
        self.layout.addStretch()

    def output_chain(self, data, sr):
        if PEDALBOARD_AVAILABLE:
            for v in self.vst_outputs:
                data = v.process(data, sr)
        return data

    def input_chain(self, data, sr):
        if PEDALBOARD_AVAILABLE:
            for v in self.vst_inputs:
                data = v.process(data, sr)
        return data

    def update_volume(self, buf):
        sample_size = buf.format().sampleSize()
        sample_count = buf.sampleCount()
        ptr = buf.constData()
        ptr.setsize(int(sample_size/8)*sample_count)

        samples = np.asarray(np.frombuffer(ptr, np.int16)).astype(float)
        rms = np.sqrt(np.mean(samples**2))
            
        level = rms / (2 ** 14)

        self.volume_meter.setValue(int(level * 100))

    def update_init_audio(self):
        if PYGAME_AVAILABLE:
            mixer.init(devicename = self.selected_dev)
            if self.mic_checkbox.isChecked():
                self.ui_parent.mic_state = True
            else:
                self.ui_parent.mic_state = False

    def update_delfiles(self):
        self.ui_parent.mic_delfiles = self.mic_output_control.isChecked()

    def set_input_dev(self, idx):
        num_audio_inputs = len(self.audio_inputs)
        if idx < num_audio_inputs:
            self.recorder.setAudioInput(self.audio_inputs[idx])

    def set_output_dev(self, idx):
        self.selected_dev = self.out_devs[idx]
        if mixer.get_init() is not None:
            mixer.quit()
            mixer.init(devicename = self.selected_dev)

    def record_dir_dialog(self):
        temp_record_dir = QFileDialog.getExistingDirectory(self,
            "Recordings Directory", self.record_dir, QFileDialog.ShowDirsOnly)
        if not os.path.exists(temp_record_dir): 
            return
        self.record_dir = temp_record_dir
        self.record_dir_label.setText(
            "Recordings directory: "+str(self.record_dir))
        
    def toggle_record(self):
        #print("toggle_record triggered at "+str(id(self)))
        if self.recorder.status() == QAudioRecorder.RecordingStatus:
            self.recorder.stop()
            self.record_button.setText("Record")
            self.last_output = self.recorder.outputLocation().toLocalFile()
            if not (PYGAME_AVAILABLE and self.mic_output_control.isChecked()):
                self.preview.from_file(self.last_output)
                self.preview.set_text("Preview - "+os.path.basename(
                    self.last_output))
            if self.automatic_checkbox.isChecked():
                self.push_to_rvc()
                self.ui_parent.convert()
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

    def push_to_rvc(self):
        if not os.path.exists(self.last_output):
            return
        self.ui_parent.clean_files = [self.last_output]
        self.ui_parent.update_file_label()
        self.ui_parent.update_input_preview()


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

class SimpleFileButton(QPushButton):
    sendFile = pyqtSignal(list)
    def __init__(self, label = "Files to Convert"):
        super().__init__(label)
        self.setAcceptDrops(True)
        self.clicked.connect(self.file_dialog)
        self.files = []

    def file_dialog(self):
        self.files = QFileDialog.getOpenFileNames(
            self, "Files to process")[0]
        self.sendFile.emit(self.files)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            self.files = []
            for url in event.mimeData().urls():
                if not url.toLocalFile():
                    continue
                self.files.append(url.toLocalFile())
            self.sendFile.emit(self.files)
            event.acceptProposedAction()
        else:
            event.ignore()
        pass

class InferenceGui(QMainWindow):
    def __init__(self, args):
        super().__init__()

        self.mic_state = False
        self.mic_delfiles = False

        self.voices = get_voices()
        self.central_widget = QFrame()
        self.layout = QHBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        self.rvc_frame = QFrame()
        self.layout.addWidget(self.rvc_frame)
        self.rvc_layout = QVBoxLayout(self.rvc_frame)

        self.output_dir = os.path.abspath("./results/")
        os.makedirs(self.output_dir, exist_ok=True)

        self.weights_box = QComboBox()
        for wt in [x["model_folder"] for x in self.voices]:
            self.weights_box.addItem(wt)
        self.weights_box.currentIndexChanged.connect(self.try_load_speaker)
        self.rvc_layout.addWidget(self.weights_box)

        self.model_state = {}
        self.hubert_model = None
        self.load_hubert()

        self.load_persist()
        self.recent_dirs = deque(
            [d for d in self.recent_dirs if os.path.exists(d)], maxlen=RECENT_DIR_MAXLEN)

        self.file_button = FileButton()
        self.clean_files = []
        self.rvc_layout.addWidget(self.file_button)
        self.file_label = QLabel("Files: "+str(self.clean_files))
        self.file_label.setWordWrap(True)
        self.rvc_layout.addWidget(self.file_label)
        self.file_button.clicked.connect(self.file_dialog) 
        self.file_button.fileDropped.connect(self.update_files) 

        self.recent_label = QLabel("Recent Directories:")
        self.rvc_layout.addWidget(self.recent_label)
        self.recent_combo = QComboBox()
        self.rvc_layout.addWidget(self.recent_combo)
        self.recent_combo.activated.connect(self.recent_dir_dialog)
        self.update_recent_combo()

        self.input_preview = AudioPreviewWidget()
        self.rvc_layout.addWidget(self.input_preview)

        self.transpose_num = QLineEdit('0')
        self.transpose_num.setValidator(QIntValidator(-36,36))
        self.transpose_frame = FieldWidget(
            QLabel("Transpose"), self.transpose_num)
        self.rvc_layout.addWidget(self.transpose_frame)

        self.f0_method_box = QComboBox()
        for x in F0_METHODS:
            self.f0_method_box.addItem(x)
        self.rvc_layout.addWidget(self.f0_method_box)

        self.index_rate_num = QLineEdit('0.0')
        self.index_rate_num.setValidator(QDoubleValidator(0.0,1.0,1))
        self.index_rate_frame = FieldWidget(
            QLabel("Index Rate"), self.index_rate_num)
        self.rvc_layout.addWidget(self.index_rate_frame)

        self.feature_search_button = SimpleFileButton(
            "Feature Search Database (*.index)")
        self.feature_search_label = QLabel("Feature search database: ")
        self.feature_search_button.sendFile.connect(
            self.write_feature_file_map)
        self.rvc_layout.addWidget(self.feature_search_button)
        self.rvc_layout.addWidget(self.feature_search_label)

        self.convert_button = QPushButton("Convert")
        self.convert_button.clicked.connect(self.convert)
        self.rvc_layout.addWidget(self.convert_button)

        self.feature_file_maps = {}

        self.output_preview = AudioPreviewWidget()
        self.rvc_layout.addWidget(self.output_preview)

        self.delete_prep_cache = []

        self.audio_recorder_and_plugins = AudioRecorderAndVSTs(self)
        self.layout.addWidget(self.audio_recorder_and_plugins)

        if len(self.voices):
            self.try_load_speaker(0)

    def recent_dir_dialog(self, index):
        # print("opening dir dialog")
        if not os.path.exists(self.recent_dirs[index]):
            print("Path did not exist: ", self.recent_dirs[index])
        self.update_files(QFileDialog.getOpenFileNames(
            self, "Files to process", self.recent_dirs[index])[0])

    def write_feature_file_map(self, userdata):
        if not "weight_path" in self.model_state:
            return
        weight_path = self.model_state["weight_path"]
        self.feature_file_maps[weight_path] = {}
        self.feature_file_maps[weight_path][
            "file_index"] = (self.feature_search_button.files[0] if
            len(self.feature_search_button.files) else None)

        self.update_feature_file_display()

    def update_feature_file_display(self):
        self.feature_search_label.setText("Feature search database: "
            + (self.feature_search_button.files[0] if
            len(self.feature_search_button.files) else ""))

    def load_feature_files(self):
        if not "weight_path" in self.model_state:
            return
        weight_path = self.model_state["weight_path"]
        print(weight_path)
        if not weight_path in self.feature_file_maps:
            return
        self.feature_search_button.files = [self.feature_file_maps[
            weight_path]["file_index"]]

    def save_persist(self):
        with open(JSON_NAME, "w") as f:
            o = {"recent_dirs": list(self.recent_dirs),
                 "output_dir": self.output_dir,
                 "feature_file_maps": self.feature_file_maps}
            json.dump(o,f)

    def load_persist(self):
        if not os.path.exists(JSON_NAME):
            self.recent_dirs = []
            self.output_dir = "./results/"
            return
        with open(JSON_NAME, "r") as f:
            o = json.load(f)
            self.recent_dirs = deque(o.get("recent_dirs",[]), maxlen=RECENT_DIR_MAXLEN)
            self.output_dir = o.get("output_dir",os.path.abspath("./results/"))
            self.feature_file_maps = o.get("feature_file_maps",{})

    def file_dialog(self):
        # print("opening file dialog")
        if not len(self.recent_dirs):
            self.update_files(QFileDialog.getOpenFileNames(
                self, "Files to process")[0])
        else:
            self.update_files(QFileDialog.getOpenFileNames(
                self, "Files to process", self.recent_dirs[0])[0])

    def update_file_label(self):
        self.file_label.setText("Files: "+str(self.clean_files))

    def update_files(self, files):
        if (files is None) or (len(files) == 0):
            return
        self.clean_files = files
        self.update_file_label()
        dir_path = os.path.abspath(os.path.dirname(self.clean_files[0]))
        if not dir_path in self.recent_dirs:
            self.recent_dirs.appendleft(dir_path)
        else:
            self.recent_dirs.remove(dir_path)
            self.recent_dirs.appendleft(dir_path)
        self.recent_combo.setCurrentIndex(self.recent_dirs.index(dir_path))
        self.update_input_preview()
        self.update_recent_combo()

    def update_recent_combo(self):
        self.recent_combo.clear()
        for d in self.recent_dirs:
            self.recent_combo.addItem(backtruncate_path(d))

    def update_input_preview(self):
        if not (PYGAME_AVAILABLE and self.mic_delfiles):
            self.input_preview.from_file(self.clean_files[0])
            self.input_preview.set_text("Preview - "+self.clean_files[0])

    def convert(self):
        outputs = self.vc_mult()
        res_paths = []

        for idx,o in enumerate(outputs):
            weight_name = (
                self.model_state["weight_path"].split(os.path.sep)[-1]
                .split('.')[0])
            sr, opt = o

            i = 1
            wav_name = Path(self.clean_files[idx]).stem
            res_path = os.path.join(self.output_dir,
                f'{wav_name}_{self.transpose_num.text()}_'
                f'{weight_name}{i}.wav')
            while os.path.exists(res_path):
                res_path = os.path.join(self.output_dir,
                    f'{wav_name}_{self.transpose_num.text()}_'
                    f'{weight_name}{i}.wav')
                i += 1

            wavfile.write(os.path.abspath(res_path), sr, opt)
            res_paths.append(res_path)

        if PYGAME_AVAILABLE and self.mic_state:
            if mixer.music.get_busy():
                mixer.music.queue(res_paths[0])
            else:
                mixer.music.load(res_paths[0])
                mixer.music.play()

        if len(res_paths) > 0 and not (PYGAME_AVAILABLE and self.mic_delfiles):
            self.output_preview.from_file(res_paths[0])
            self.output_preview.set_text("Preview - "+res_paths[0])

        if self.mic_delfiles:
            # Not sure how else to handle this without expensive loop
            self.delete_prep_cache.append(clean_name)
            self.delete_prep_cache.append(wav_path)
            self.delete_prep_cache.append(res_path)
            self.try_delete_prep_cache()

        return outputs

    def try_delete_prep_cache(self):
        for f in self.delete_prep_cache:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    self.delete_prep_cache.remove(f)
                except PermissionError as e:
                    continue

    def vc_mult(self, sid=0):
        outputs = []
        for f in self.clean_files:
            info, opt_tup = self.vc_single(f, sid)
            outputs.append(opt_tup)
        return outputs

    def vc_single(self, input_audio, sid=0):  
        f0_up_key = int(self.transpose_num.text()) 
        try:
            audio = load_audio(input_audio, 16000)
            times = [0, 0, 0]
            if self.hubert_model is None:
                load_hubert()
            if_f0 = self.model_state["cpt"].get("f0", 1) # why this fukt?
            if not len(self.feature_search_button.files):
                file_index = ""
                file_big_npy = ""
            else:
                file_index = self.feature_search_button.files[0]
                file_index = file_index.strip(" ").strip('"').strip("\n"
                    ).strip('"').strip(" ").replace("trained","added") 
            audio_opt = self.model_state["vc"].pipeline(
                self.hubert_model, # model
                self.model_state["net_g"], # net_g
                sid, # sid
                audio, # audio
                input_audio, #input_audio_path
                times, # times
                f0_up_key, # f0_up_key
                F0_METHODS[self.f0_method_box.currentIndex()], # f0_method
                file_index, # file_index
                #file_big_npy, 
                float(self.index_rate_num.text()), # index_rate
                if_f0, # if_f0
                filter_radius = 3, # TODO this is a harvest setting?
                tgt_sr = self.model_state["tgt_sr"],
                resample_sr = self.model_state["tgt_sr"],
                rms_mix_rate = 1, # TODO this uses the input as a volume?
                version = "v2", # TODO how do you distnguish between 2 model vs
                protect = 0.0, # TODO
                f0_file=None, 
            )
            #print(audio_opt)
            #print(
                #"npy: ", times[0], "s, f0: ", times[1], "s, infer: ",
                #times[2], "s", sep=""
            #)
            return "Success", (self.model_state["tgt_sr"], audio_opt)
        except:
            info = traceback.format_exc()
            print(info)
            return info, (None, None)

    def load_hubert(self):
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            ["hubert_base.pt"],
            suffix="",
        )
        hubert_model = models[0]
        hubert_model = hubert_model.to(config.device)
        if config.is_half:
            hubert_model = hubert_model.half()
        else:
            hubert_model = hubert_model.float()
        hubert_model.eval()
        self.hubert_model = hubert_model

    def try_load_speaker(self, idx):
        cpt = torch.load(
            os.path.join(self.voices[idx]["weight_path"]),
            map_location="cpu")
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        n_spk = cpt["config"][-3]
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")

        if version == "v1":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(*cpt["config"],
                    is_half=config.is_half)
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        elif version == "v2":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs768NSFsid(*cpt["config"],
                    is_half=config.is_half)
            else:
                net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
        del net_g.enc_q
        # I love it when people put statements inside prints that actually
        # change something
        netg_print = net_g.load_state_dict(cpt["weight"], strict=False)
        print(netg_print)  

        net_g.eval().to(config.device)

        if config.is_half:
            net_g = net_g.half()
        else:
            net_g = net_g.float()

        self.model_state["weight_path"] = self.voices[idx]["weight_path"]
        self.model_state["cpt"] = cpt
        self.model_state["tgt_sr"] = tgt_sr
        self.model_state["net_g"] = net_g
        self.model_state["vc"] = VC(tgt_sr, config)
        self.model_state["n_spk"] = n_spk

        self.feature_file_maps[self.voices[idx]["weight_path"]] = {}
        self.feature_file_maps[ 
            self.voices[idx]["weight_path"]]["file_index"] = (
            self.voices[idx]["feature_index"])

        self.load_feature_files()
        self.update_feature_file_display()

if __name__ == "__main__":
    import argparse
    app = QApplication(sys.argv)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    w = InferenceGui(args)
    w.show()
    app.exec()
    w.save_persist()
