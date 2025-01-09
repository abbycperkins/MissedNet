import pandas as pd
import librosa
import sounddevice as sd
import sys
import matplotlib
from pandas import DataFrame
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import webbrowser
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QApplication, QVBoxLayout, QLabel, QPushButton, QMessageBox, QFileDialog, QComboBox, qApp,
    QGridLayout, QSizePolicy, QProgressBar
)
from datetime import datetime
from pathlib import Path


class Worker(QtCore.QObject):
    count_changed = QtCore.pyqtSignal(int)
    loading_complete = QtCore.pyqtSignal(dict)

    def __init__(self, files, path):
        super().__init__()
        self.files = files
        self.path = path

    def load_files(self):
        total_files = len(self.files)
        audio_files = {}
        for index, file in enumerate(self.files):
            wav_path = list(self.path.glob(f'*{file}*'))
            audio_files[file] = librosa.load(wav_path[0])
            progress = int((index + 1) / total_files * 100)
            self.count_changed.emit(progress)
        self.loading_complete.emit(audio_files)

class PopUpProgressBar(QWidget): # TODO make main parent of this so the progress bar is in the right place
    loading_complete = QtCore.pyqtSignal(dict)

    def __init__(self, files, path):
        super().__init__()
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 500, 75)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.pbar)
        self.setLayout(self.layout)

        self.setGeometry(300, 300, 550, 100)
        self.setWindowTitle('Loading Files')

        # Thread and Worker setup
        self.thread = QtCore.QThread()
        self.worker = Worker(files, path)
        self.worker.moveToThread(self.thread)

        # Signal-slot connections
        self.worker.count_changed.connect(self.on_count_changed)
        self.worker.loading_complete.connect(self.loading_complete.emit)
        self.worker.loading_complete.connect(self.on_loading_complete)
        self.thread.started.connect(self.worker.load_files)

        self.loop = None

    def start_progress(self):
        self.show()
        self.thread.start()
        self.loop = QtCore.QEventLoop()
        self.loading_complete.connect(self.loop.quit)
        self.loop.exec_() # needs to wait until all files are loaded, then continue with the program

    @QtCore.pyqtSlot(int)
    def on_count_changed(self, value):
        self.pbar.setValue(value)

    @QtCore.pyqtSlot(dict)
    def on_loading_complete(self):
        self.thread.quit()
        self.thread.wait()

class SoundThread(QtCore.QThread):
    def __init__(self, y, sr):
        super().__init__()
        self.y = y
        self.sr = sr
        print(f'{self.y}:{self.sr}')

    def run(self):
        sd.wait()
        sd.play(self.y, self.sr)
        sd.wait()

class Settings(QMainWindow):
    runAnalysis = QtCore.pyqtSignal(Path, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.time = None
        self.file_path = None

        # dropdown to select timeframe
        self.times = QComboBox()
        self.times.addItem('Select Timeframe of Interest')
        self.times.addItem('Day')
        self.times.addItem('Week')
        self.times.addItem('Month')
        lay = QVBoxLayout()

        lay.addWidget(self.times)
        self.times.activated.connect(self.current_text)

        # button to select directory and start
        start = QPushButton('Select Directory and Start Analysis')
        lay.addWidget(start)
        start.pressed.connect(lambda: self.load_location())

        starting_widget = QWidget()
        starting_widget.setLayout(lay)
        self.setCentralWidget(starting_widget)

    def current_text(self, _):
        self.time = self.times.currentText()

    def load_location(self):
        self.file_path = Path(str(QFileDialog.getExistingDirectory(self, 'Select Directory')))
        self.runAnalysis.emit(self.file_path, self.time)

class AudioAnalysis(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Analysis")
        self.setFixedSize(600, 450)
        layout = QGridLayout()

        self.settings = Settings(self)
        self.settings.runAnalysis.connect(self.run_analysis)

        self.image_label = QLabel()
        layout.addWidget(self.image_label, 0, 0, 1, 2, Qt.AlignHCenter)

        self.species_label = QLabel("Species Name")
        self.species_label.setStyleSheet("font-size: 24px; font-weight: bold;")

        layout.addWidget(self.species_label, 1, 0, 1, 2, Qt.AlignHCenter)

        self.goodID = QPushButton('Good Identification')
        self.goodID.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.goodID.setStyleSheet('font-size: 16px; font-weight: bold; background-color: #D0D991') # green
        self.badID = QPushButton('Not Confirmed')
        self.badID.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.badID.setStyleSheet('font-size: 16px; font-weight: bold; background-color: #F2D5CE') # red
        self.repeat = QPushButton('Replay Clip')
        self.repeat.setStyleSheet('font-size: 14px')
        self.open_web = QPushButton('Open Audio Examples')
        self.open_web.setStyleSheet('font-size: 14px')

        layout.addWidget(self.goodID, 3, 0, 1, 1)
        layout.addWidget(self.badID, 3, 1, 1, 1)
        layout.addWidget(self.repeat, 2, 0, 1, 1)
        layout.addWidget(self.open_web, 2, 1, 1, 1)

        self.goodID.pressed.connect(lambda: self.next_sound(True))
        self.badID.pressed.connect(lambda: self.next_sound(False))
        self.repeat.pressed.connect(lambda: self.play_sound())
        self.open_web.pressed.connect(lambda: self.open_website())

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.output: DataFrame | None = None
        self.file_path: Path | None = None
        self.time: str | None = None
        self.detection: list | None = None
        self.counter = 0
        self.species_counter = 0
        self.period_counter = -1
        self.audio_files: dict | None = None
        self.selection_df_final: DataFrame | None = None
        self.species_list: list | None = None
        self.species_detections: DataFrame | None = None
        self.df: DataFrame | None = None
        self.data_path: Path | None = None
        self.selection_path: Path | None = None
        self.sr: int | None = None
        self.part: np.ndarray | None = None

    def start(self) -> None:
        self.settings.show()

    def open_website(self):
        species = self.detection[8].replace("'", "").replace(" ", "_")
        url = f'https://www.allaboutbirds.org/guide/{species}/sounds'
        webbrowser.open(url)

    def first_sound(self):
        self.disable_buttons()
        self.species_counter = 0
        self.species_detections = self.selection_df_final[
            self.selection_df_final['Label'] == self.species_list[self.species_counter]
        ].sort_values(by='Score', ascending=False)

        self.detection = self.species_detections.iloc[self.counter]

        # Update species label
        self.species_label.setText(self.detection[8])

        self.play_sound()

    def next_sound(self, good_id):
        self.disable_buttons()
        if good_id:
            self.output.at[self.period_counter, self.detection[8]] = 'Confirmed Present'
            # move on to next species
            self.species_counter = self.species_counter + 1
            self.counter = 0
        if not good_id:
            self.counter = self.counter + 1
            if self.counter == self.species_detections.shape[0]:
                self.output.at[self.period_counter, self.detection[8]] = 'Failed Verification'
                self.species_counter = self.species_counter + 1
                self.counter = 0
        if self.species_counter == len(self.species_list):
            self.initialize_period()
        else:
            self.species_detections = self.selection_df_final[
                self.selection_df_final['Label'] == self.species_list[self.species_counter]
            ].sort_values(by='Score', ascending=False)

            self.detection = self.species_detections.iloc[self.counter]

            # Update species label
            self.species_label.setText(self.detection[8])

            self.play_sound()

    def plot_spec(self):
        d = librosa.amplitude_to_db(np.abs(librosa.stft(self.part, n_fft=512)), ref=np.max)
        plt.figure(figsize=(5, 3))
        librosa.display.specshow(d, y_axis='linear', sr=self.sr, x_axis='time', cmap = 'gist_yarg')
        plt.colorbar(format="%+2.f dB")
        plt.tight_layout()
        plt.savefig("temp_image.png", bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
        plt.close()

        pixmap = QPixmap("temp_image.png")
        scaled_pixmap = pixmap.scaled(500, 300, transformMode=Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        # qApp.processEvents()

    def play_sound(self):
        self.disable_buttons()
        start_time = self.detection[3]
        end_time = self.detection[4]
        y, self.sr = self.audio_files[self.detection[9]]
        self.part = y[int(start_time * self.sr):int(end_time * self.sr)]

        self.plot_spec()

        self.worker = SoundThread(self.part, self.sr)
        self.worker.start()
        self.worker.finished.connect(self.activate_buttons)

    def activate_buttons(self):
        self.goodID.setEnabled(True)
        self.badID.setEnabled(True)
        # self.open_web.setEnabled(True)
        self.repeat.setEnabled(True)

    def disable_buttons(self):
        self.goodID.setEnabled(False)
        self.badID.setEnabled(False)
        # self.open_web.setEnabled(False)
        self.repeat.setEnabled(False)

    @QtCore.pyqtSlot(Path, str)
    def run_analysis(self, f, t):
        self.settings.hide()
        self.show()
        self.file_path = f
        self.time = t

        self.selection_path = self.file_path / 'Selections'
        self.data_path = self.file_path / 'Data'

        date_list = []
        all_paths = list(self.data_path.glob('*'))
        all_files = [i.name for i in all_paths]
        names_only = [i.replace('.wav', '') for i in all_files]
        all_files_glob = [f'*{i}*' for i in names_only]

        self.df = pd.DataFrame(names_only, columns=['PATH'])

        for i in all_paths:
            date_list.append(datetime.strptime(i.as_posix().split(sep='_')[-2],'%Y%m%d'))

        if self.time == 'Day':
            self.df['Date'] = date_list
            self.output = self.df.drop_duplicates(subset='Date')
            self.output = self.output.reset_index(drop=True)

        if self.time == 'Week':
            week_list = []
            for x in date_list:
                week_list.append(x.strftime('%U-%Y'))
            self.df['Date'] = week_list
            self.output = self.df.drop_duplicates(subset='Date')
            self.output = self.output.reset_index(drop=True)

        if self.time == 'Month':
            month_list = []
            for x in date_list:
                month_list.append(x.strftime('%b-%Y'))
            self.df['Date'] = month_list
            self.output = self.df.drop_duplicates(subset='Date')
            self.output = self.output.reset_index(drop=True)

        self.initialize_period()

    def initialize_period(self):
        # back up every period
        self.output.drop(columns = 'PATH').to_csv(self.file_path / 'results.csv', index=False)

        # tick counter
        self.period_counter = self.period_counter + 1 # starts at 0
        if self.period_counter == self.output.shape[0]: # check if last period completed
            self.close()
            msg = QMessageBox()
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setText('Audio Analysis completed! Quit and view your output file.')
            msg.setWindowTitle('Complete!')
            msg.exec_()
        else:
            selection_df_list = []
            self.audio_files = {}

            # Find the selection files
            period_files = self.df[self.df['Date'] == self.output.iloc[self.period_counter]['Date']]
            for file in period_files['PATH']:
                selection_df= pd.read_csv(list(self.selection_path.glob(f'*{file}*'))[0], delimiter='\t')
                selection_df['File'] = file
                selection_df_list.append(selection_df)

            selection_df = pd.concat(selection_df_list, ignore_index=True)
            self.selection_df_final = []
            if 'Label' in selection_df.columns:
                for index, row in selection_df.iterrows():
                    if not row['Label'].islower(): # weird bug in Raven Pro where sometimes species names are abbreviated ex: baleag == Bald Eagle
                        self.selection_df_final.append(row)
                self.selection_df_final = pd.DataFrame(self.selection_df_final)
                self.species_list = self.selection_df_final.loc[:, 'Label'].unique()

                for spec in self.species_list:
                    if 'spec' in self.output.columns:
                        continue
                    else: self.output[spec] = ''

                popup = PopUpProgressBar(period_files['PATH'], self.data_path)
                popup.loading_complete.connect(self.receive_dict)
                popup.start_progress()
                self.first_sound()
            else:
                self.initialize_period()

    @QtCore.pyqtSlot(dict)
    def receive_dict(self, d):
        self.audio_files = d

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = AudioAnalysis()
    main.start()
    sys.exit(app.exec_())