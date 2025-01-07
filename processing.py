import pandas as pd
import librosa
import sounddevice as sd
import sys
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QApplication, QVBoxLayout, QTreeWidget, QLabel, QDialogButtonBox, QFormLayout,
    QTreeWidgetItem, QPushButton, QMessageBox, QFileDialog, QLineEdit, QDialog, QComboBox, qApp
)
from datetime import datetime
from pathlib import Path


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
        plt.ioff()
        self.setWindowTitle("Audio Analysis")
        self.setFixedSize(600, 400)
        layout = QVBoxLayout()

        self.settings = Settings(self)
        self.settings.runAnalysis.connect(self.run_analysis)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.species_label = QLabel("Species Name")
        self.species_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        layout.addWidget(self.species_label)
        self.species_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        goodID = QPushButton('Good Identification')
        badID = QPushButton('Not Confirmed')
        repeat = QPushButton('Replay Clip')

        layout.addWidget(goodID)
        layout.addWidget(badID)
        layout.addWidget(repeat)

        goodID.pressed.connect(lambda: self.goodID_next_sound())
        badID.pressed.connect(lambda: self.badID_next_sound())
        repeat.pressed.connect(lambda: self.play_sound_again())

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.output = None
        self.file_path = None
        self.time = None
        self.detection = None
        self.counter = 0
        self.species_counter = 0
        self.audio_files = None
        self.selection_df_final = None
        self.species_list = None
        self.species_detections = None

    def start(self) -> None:
        self.settings.show()

    def first_sound(self):
        self.species_detections = self.selection_df_final[self.selection_df_final['Label'] == self.species_list[self.species_counter]].sort_values(by='Score', ascending=False)

        self.detection = self.species_detections.iloc[self.counter]

        # Update species label
        self.species_label.setText(self.detection[8])

        start_time = self.detection[3]
        end_time = self.detection[4]
        y, sr = self.audio_files[self.detection[9]]
        part = y[int(start_time * sr):int(end_time * sr)]

        # Update QPixmap

        hop_length = 1024
        D = librosa.amplitude_to_db(np.abs(librosa.stft(part, hop_length=hop_length)),
                                    ref=np.max)
        plt.figure(figsize=(4, 2))
        librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time')

        plt.tight_layout()
        plt.savefig("temp_image.png", bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        pixmap = QPixmap("temp_image.png")
        scaled_pixmap = pixmap.scaled(300, 200)
        self.image_label.setPixmap(scaled_pixmap)

        qApp.processEvents()
        sd.wait()
        sd.play(part, sr)
        sd.wait()

    def goodID_next_sound(self):
        # move on to next species
        self.species_counter = self.species_counter + 1
        self.counter = 0

        if self.species_counter == len(self.species_list):
            msg = QMessageBox()
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setText('Audio Analysis completed! Quit and view your output file.')
            msg.setWindowTitle('Complete!')
            msg.exec_()
        else:
            self.species_detections = self.selection_df_final[self.selection_df_final['Label'] == self.species_list[self.species_counter]].sort_values(by='Score', ascending=False)

        self.detection = self.species_detections.iloc[self.counter]

        # Update species label
        self.species_label.setText(self.detection[8])

        start_time = self.detection[3]
        end_time = self.detection[4]
        y, sr = self.audio_files[self.detection[9]]
        part = y[int(start_time * sr):int(end_time * sr)]

        # Update QPixmap

        hop_length = 1024
        D = librosa.amplitude_to_db(np.abs(librosa.stft(part, hop_length=hop_length)),
                                    ref=np.max)
        plt.figure(figsize=(4, 2))
        librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time')

        plt.tight_layout()
        plt.savefig("temp_image.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        pixmap = QPixmap("temp_image.png")
        scaled_pixmap = pixmap.scaled(300, 200)
        self.image_label.setPixmap(scaled_pixmap)
        qApp.processEvents()

        sd.wait()
        sd.play(part, sr)
        sd.wait()


    def badID_next_sound(self):
        self.counter = self.counter + 1
        if self.counter == self.species_detections.shape[0]:
            self.species_counter = self.species_counter + 1
            self.counter = 0
            self.species_detections = self.selection_df_final[
                self.selection_df_final['Label'] == self.species_list[self.species_counter]].sort_values(by='Score',
                                                                                                         ascending=False)
            if self.species_counter == len(self.species_list):
                msg = QMessageBox()
                msg.setStandardButtons(QMessageBox.Ok)
                msg.setText('Audio Analysis completed! Quit and view your output file.')
                msg.setWindowTitle('Complete!')
                msg.exec_()
            self.detection = self.species_detections.iloc[self.counter]

            # Update species label
            self.species_label.setText(self.detection[8])
        else:
            self.detection = self.species_detections.iloc[self.counter]

        start_time = self.detection[3]
        end_time = self.detection[4]
        y, sr = self.audio_files[self.detection[9]]
        part = y[int(start_time * sr):int(end_time * sr)]

        # Update QPixmap

        hop_length = 1024
        D = librosa.amplitude_to_db(np.abs(librosa.stft(part, hop_length=hop_length)),
                                    ref=np.max)
        plt.figure(figsize=(4, 2))
        librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time')

        plt.tight_layout()
        plt.savefig("temp_image.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        pixmap = QPixmap("temp_image.png")
        scaled_pixmap = pixmap.scaled(300, 200)
        self.image_label.setPixmap(scaled_pixmap)
        qApp.processEvents()

        sd.wait()
        sd.play(part, sr)
        sd.wait()

    def play_sound_again(self):
        start_time = self.detection[3]
        end_time = self.detection[4]
        y, sr = self.audio_files[self.detection[9]]
        part = y[int(start_time * sr):int(end_time * sr)]
        qApp.processEvents()

        sd.wait()
        sd.play(part, sr)
        sd.wait()

    @QtCore.pyqtSlot(Path, str)
    def run_analysis(self, f, t):
        self.settings.hide()
        self.show()
        self.file_path = f
        self.time = t

        selection_path = self.file_path / 'Selections'
        data_path = self.file_path / 'Data'

        date_list = []
        all_paths = list(data_path.glob('*'))
        all_files = [i.name for i in all_paths]
        names_only = [i.replace('.wav', '') for i in all_files]
        all_files_glob = [f'*{i}*' for i in names_only]

        df = pd.DataFrame(names_only, columns=['PATH'])

        for i in all_paths:
            date_list.append(datetime.strptime(i.as_posix().split(sep='_')[-2],'%Y%m%d'))

        if self.time == 'Day':
            df['Date'] = date_list
            self.output = df.drop_duplicates(subset='Date')

        if self.time == 'Week':
            week_list = []
            for x in date_list:
                week_list.append(x.strftime('%U-%Y'))
            df['Date'] = week_list
            self.output = df.drop_duplicates(subset='Date')

        if self.time == 'Month':
            month_list = []
            for x in date_list:
                month_list.append(x.strftime('%b-%Y'))
            df['Date'] = month_list
            self.output = df.drop_duplicates(subset='Date')

        for period in self.output['Date']:
            selection_df_list = []
            self.audio_files = {}

            # Find the selection files
            period_files = df[df['Date'] == period]

            for file in period_files['PATH']:
                selection_df= pd.read_csv(list(selection_path.glob(f'*{file}*'))[0], delimiter='\t')
                selection_df['File'] = file
                selection_df_list.append(selection_df)

            selection_df = pd.concat(selection_df_list, ignore_index=True)
            self.selection_df_final = []

            for index, row in selection_df.iterrows():
                if not row['Label'].islower():
                    self.selection_df_final.append(row)
            self.selection_df_final = pd.DataFrame(self.selection_df_final)
            self.species_list = self.selection_df_final['Label'].unique()

            for file in period_files['PATH']:
                wav_path = list(data_path.glob(f'*{file}*'))
                self.audio_files[file] = librosa.load(wav_path[0])
            self.first_sound()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = AudioAnalysis()
    main.start()
    sys.exit(app.exec_())