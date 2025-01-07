import pandas as pd
import librosa
import sounddevice as sd
import sys
import matplotlib.pyplot as plt
import numpy as np
import webbrowser
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
        open_web = QPushButton('Open Audio Examples')

        layout.addWidget(goodID)
        layout.addWidget(badID)
        layout.addWidget(repeat)
        layout.addWidget(open_web)

        goodID.pressed.connect(lambda: self.goodID_next_sound())
        badID.pressed.connect(lambda: self.badID_next_sound())
        repeat.pressed.connect(lambda: self.play_sound_again())
        open_web.pressed.connect(lambda: self.open_website())

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.output = None
        self.file_path: Path | None = None # TODO add this to all plz
        self.time = None
        self.detection = None
        self.counter = 0
        self.species_counter = 0
        self.period_counter = -1
        self.audio_files = None
        self.selection_df_final = None
        self.species_list = None
        self.species_detections = None
        self.df = None
        self.data_path = None
        self.selection_path = None

    def start(self) -> None:
        self.settings.show()

    def open_website(self):
        species = self.detection[8].replace("'", "").replace(" ", "_")
        url = f'https://www.allaboutbirds.org/guide/{species}/sounds'
        webbrowser.open(url)

    def first_sound(self):
        self.species_detections = self.selection_df_final[
            self.selection_df_final['Label'] == self.species_list[self.species_counter]
        ].sort_values(by='Score', ascending=False)

        self.detection = self.species_detections.iloc[self.counter]

        # Update species label
        self.species_label.setText(self.detection[8])

        start_time = self.detection[3]
        end_time = self.detection[4]
        y, sr = self.audio_files[self.detection[9]]
        part = y[int(start_time * sr):int(end_time * sr)]

        # Update QPixmap

        D = librosa.amplitude_to_db(np.abs(librosa.stft(part)), ref=np.max)
        plt.figure(figsize=(4, 2))
        librosa.display.specshow(D, y_axis='linear', sr=sr, x_axis='time')

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
        self.output.at[self.period_counter, self.detection[8]] = 'Confirmed Present'
        # move on to next species
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

        start_time = self.detection[3]
        end_time = self.detection[4]
        y, sr = self.audio_files[self.detection[9]]
        part = y[int(start_time * sr):int(end_time * sr)]

        # Update QPixmap

        D = librosa.amplitude_to_db(np.abs(librosa.stft(part)), ref=np.max)
        plt.figure(figsize=(4, 2))
        librosa.display.specshow(D, y_axis='linear', sr=sr, x_axis='time')

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


    def badID_next_sound(self):
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
        else:
            self.detection = self.species_detections.iloc[self.counter]

        start_time = self.detection[3]
        end_time = self.detection[4]
        y, sr = self.audio_files[self.detection[9]]
        part = y[int(start_time * sr):int(end_time * sr)]

        # Update QPixmap

        D = librosa.amplitude_to_db(np.abs(librosa.stft(part)), ref=np.max)
        plt.figure(figsize=(4, 2))
        librosa.display.specshow(D, y_axis='linear', sr=sr, x_axis='time')

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
        self.output.to_csv(self.file_path / 'results.csv', index=False)

        # tick counter
        self.period_counter = self.period_counter + 1 # starts at 0
        if self.period_counter == self.output.shape[0]: # check if last period completed
            self.close()
            msg = QMessageBox()
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setText('Audio Analysis completed! Quit and view your output file.')
            msg.setWindowTitle('Complete!')
            msg.exec_()
            # TODO make this close everything
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
                    else: self.output[spec] = 'Not Detected'

                for file in period_files['PATH']:
                    wav_path = list(self.data_path.glob(f'*{file}*'))
                    self.audio_files[file] = librosa.load(wav_path[0])

                self.first_sound()
            else:
                self.initialize_period()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = AudioAnalysis()
    main.start()
    sys.exit(app.exec_())