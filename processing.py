import pandas as pd
import librosa
import sounddevice as sd
import sys
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QApplication, QVBoxLayout, QTreeWidget, QLabel, QDialogButtonBox, QFormLayout,
    QTreeWidgetItem, QPushButton, QMessageBox, QFileDialog, QLineEdit, QDialog, QComboBox,
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
        self.setWindowTitle("Audio Analysis")
        layout = QVBoxLayout()

        self.settings = Settings(self)
        self.settings.runAnalysis.connect(self.run_analysis)

        goodID = QPushButton('Good Identification')
        badID = QPushButton('Not Confirmed')
        repeat = QPushButton('Replay Clip')

        layout.addWidget(goodID)
        layout.addWidget(badID)
        layout.addWidget(repeat)

        goodID.pressed.connect(lambda: self.play_sound())
        badID.pressed.connect(lambda: self.play_sound())
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

    def start(self) -> None:
        self.settings.show()

    def firstsound(self):
        self.species_list = self.selection_df_final[self.selection_df_final['Label'] == self.species_list[self.species_counter]].sort_values(by='Score', ascending=False)

        # TODO show species somewhere
        self.detection = self.species_list.iloc[self.counter]
        start_time = self.detection[3]
        end_time = self.detection[4]
        y, sr = self.audio_files[self.detection[9]]
        part = y[int(start_time * sr):int(end_time * sr)]

        sd.wait()
        sd.play(part, sr)
        sd.wait()

    def goodID_next_sound(self):
        start_time = self.detection[4]
        end_time = self.detection[5]
        y, sr = self.audio_files[self.detection[10]]
        part = y[int(start_time * sr):int(end_time * sr)]

        # Next species
        # TODO first sound

        sd.wait()
        sd.play(part, sr)
        sd.wait()


    def badID_next_sound(self):
        start_time = self.detection[4]
        end_time = self.detection[5]
        y, sr = self.audio_files[self.detection[10]]
        part = y[int(start_time * sr):int(end_time * sr)]

        sd.wait()
        sd.play(part, sr)
        sd.wait()
        self.counter = self.counter + 1

    def play_sound_again(self):
        start_time = self.detection[4]
        end_time = self.detection[5]
        y, sr = self.audio_files[self.detection[10]]
        part = y[int(start_time * sr):int(end_time * sr)]

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

        if self.time == 'Year':
            year_list = []
            for x in date_list:
                year_list.append(x.strftime('%Y'))
            df['Date'] = year_list
            self.output = df.drop_duplicates(subset='Date')

        for period in self.output['Date']:
            selection_df_list = []
            self.audio_files = {}
            txt_path = []

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
        self.firstsound()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = AudioAnalysis()
    main.start()
    sys.exit(app.exec_())