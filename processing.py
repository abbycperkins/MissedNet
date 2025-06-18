import pandas as pd
import librosa
import sounddevice as sd
import soundfile as sf
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import webbrowser
import re
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QApplication, QVBoxLayout, QLabel, QPushButton, QMessageBox, QFileDialog, QComboBox, qApp,
    QGridLayout, QSizePolicy, QProgressBar, QStatusBar, QSlider, QDockWidget, QRadioButton, QButtonGroup, QAction,
    QHBoxLayout, QDialog
)
from datetime import datetime, timedelta
from pathlib import Path
from pandas import DataFrame


class SoundThread(QtCore.QThread):
    def __init__(self, y, sr, vol):
        super().__init__()
        self.y = y * vol * 6
        self.sr = sr

    def run(self):
        sd.wait()
        sd.play(self.y, self.sr)
        sd.wait()

class Settings(QDialog):
    runAnalysis = QtCore.pyqtSignal(Path, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.time = None
        self.file_path = None
        self.setWindowTitle('Load Files')
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

        self.setLayout(lay)

    def current_text(self, _):
        self.time = self.times.currentText()

    def load_location(self):
        self.file_path = Path(str(QFileDialog.getExistingDirectory(self, 'Select Directory')))
        self.runAnalysis.emit(self.file_path, self.time)

class AudioAnalysis(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MissedNET: Audio Analysis")
        self.setWindowIcon(QIcon('window_icon.png'))
        self.setFixedSize(600, 550)
        layout = QGridLayout()

        self.settings = Settings(self)
        self.settings.runAnalysis.connect(self.run_analysis)

        self.image_label = QLabel()
        layout.addWidget(self.image_label, 0, 0, 1, 7, Qt.AlignHCenter)
        pixmap = QPixmap("logo.png")
        self.image_label.setPixmap(pixmap)

        arrow_style = """
            font-size: 36px;               /* Make triangle larger */
            font-weight: bold;
            min-width: 30px;               /* Make button wider */
            min-height: 30px;              /* Make button taller */
            padding: 10px;
        """

        # Create arrow buttons
        self.left_arrow = QPushButton("◀")
        self.left_arrow.setFixedWidth(40)
        self.left_arrow.setStyleSheet(arrow_style)
        self.right_arrow = QPushButton("▶")
        self.right_arrow.setFixedWidth(40)
        self.right_arrow.setStyleSheet(arrow_style)
        # Link to functions
        self.left_arrow.clicked.connect(lambda: self.next_sound(new_species=False, forward=False))
        self.right_arrow.clicked.connect(lambda: self.next_sound(new_species=False, forward=True))

        self.export_clip = QPushButton("Export Clip")
        self.export_clip.setStyleSheet('font-size: 14px')
        layout.addWidget(self.export_clip, 3, 0, 1, 2)
        self.export_clip.clicked.connect(self.export_current_clip)

        # Create the species label
        self.species_label = QLabel("Species Name")
        self.species_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.species_label.setAlignment(Qt.AlignCenter)

        # Combine into horizontal layout
        species_layout = QHBoxLayout()
        species_layout.addWidget(self.left_arrow)
        species_layout.addWidget(self.species_label, stretch=1)
        species_layout.addWidget(self.right_arrow)

        # Add the layout to the grid
        species_widget = QWidget()
        species_widget.setLayout(species_layout)
        layout.addWidget(species_widget, 1, 0, 1, 6)

        self.info_label = QLabel("Score 0.### | Detection #/#")
        self.info_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        layout.addWidget(self.info_label, 2, 0, 1, 6, Qt.AlignHCenter)

        self.goodID = QPushButton('Good Identification')
        self.goodID.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.goodID.setStyleSheet('font-size: 16px; font-weight: bold; background-color: #8ab1ff')
        self.badID = QPushButton('Not Confirmed')
        self.badID.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.badID.setStyleSheet('font-size: 16px; font-weight: bold; background-color: #F2D5CE') # red
        self.repeat = QPushButton('Replay Clip')
        self.repeat.setStyleSheet('font-size: 14px')
        self.open_web = QPushButton('Open Audio Examples')
        self.open_web.setStyleSheet('font-size: 14px')

        layout.addWidget(self.goodID, 4, 0, 1, 3)
        layout.addWidget(self.badID, 4, 3, 1, 3)
        layout.addWidget(self.repeat, 3, 2, 1, 2)
        layout.addWidget(self.open_web, 3, 4, 1, 2)

        self.goodID.pressed.connect(lambda: self.detection_decision(good_id=True))
        self.badID.pressed.connect(lambda: self.detection_decision(good_id=False))
        self.repeat.pressed.connect(lambda: self.play_sound())
        self.open_web.pressed.connect(lambda: self.open_website())

        self.status = QStatusBar()
        self.status.setSizeGripEnabled(False)

        layout.addWidget(self.status, 5, 0, 1, 7)

        self.dock = QDockWidget('Advanced Options')
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        self.dock_content = QWidget()
        self.dock_layout = QVBoxLayout(self.dock_content)
        self.dock.setWidget(self.dock_content)

        self.dock.setFloating(True)
        self.dock.setGeometry(self.geometry().right() + 10, self.geometry().top(), 200, 150)

        self.dock.hide()

        self.toggle_dock_action = QAction("Show Advanced Options", self)
        self.toggle_dock_action.setCheckable(True)
        self.toggle_dock_action.triggered.connect(self.toggle_dock)
        self.dock.visibilityChanged.connect(self.update_toggle_button)
        self.menuBar().addAction(self.toggle_dock_action)

        self.dock_buttons = QButtonGroup()

        self.default = QRadioButton('Default: check every period')
        self.default.setChecked(True)

        self.black = QRadioButton('Blacklist: assume never present')

        self.gray = QRadioButton('Graylist: reduce to weekly checks')

        self.white = QRadioButton('Whitelist: assume always present')

        self.default.setObjectName("default")
        self.black.setObjectName("blacklist")
        self.gray.setObjectName("graylist")
        self.white.setObjectName("whitelist")

        self.dock_buttons.addButton(self.default)
        self.dock_buttons.addButton(self.white)
        self.dock_buttons.addButton(self.gray)
        self.dock_buttons.addButton(self.black)

        # Add individual buttons to the layout
        self.dock_layout.addWidget(self.default)
        self.dock_layout.addWidget(self.black)
        self.dock_layout.addWidget(self.gray)
        self.dock_layout.addWidget(self.white)

        self.progress = QProgressBar()

        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                background-color: #f0f0f0;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #5589aa;
                width: 10px;
            }
        """)

        self.message = QLabel('Loading Audio Files...')
        self.message.setStyleSheet("font-size: 12px;")
        self.status.addPermanentWidget(self.progress, stretch = 1)
        self.status.addWidget(self.message)

        vol_layout = QVBoxLayout()

        self.volume = QSlider()
        self.volume.setFocusPolicy(Qt.NoFocus)
        self.volume.setValue(20)
        self.volume.setFixedWidth(30)
        self.volume.setStyleSheet("QSlider::handle:horizontal {background-color: #5488ff;}")

        vol_label = QLabel()
        pixmap=QPixmap("volume.png")
        scaled_pixmap = pixmap.scaled(int(284/6), int(162/6), transformMode=Qt.SmoothTransformation)
        vol_label.setPixmap(scaled_pixmap)
        vol_layout.addWidget(vol_label)
        vol_layout.addWidget(self.volume, alignment = Qt.AlignHCenter)

        layout.addLayout(vol_layout, 1, 6, 4, 1)

        for i in range(6):
            layout.setColumnStretch(i, 6)
        layout.setColumnStretch(6, 1)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.disable_buttons()

        self.output: DataFrame | None = None
        self.file_path: Path | None = None
        self.time: str | None = None
        self.detection: DataFrame | None = None
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
        self.behavior: dict[str, str] = {}


    def start(self) -> None:
        self.show()

        self.settings.show()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Left and self.left_arrow.isEnabled():
            self.left_arrow.click()
        elif key == Qt.Key_Right and self.right_arrow.isEnabled():
            self.right_arrow.click()
        elif key in (Qt.Key_Enter, Qt.Key_Return) and self.goodID.isEnabled():
            self.goodID.click()
        elif key == Qt.Key_Backspace and self.badID.isEnabled():
            self.badID.click()

    def position_dock(self):
        main_geom = self.geometry()
        dock_width = 200
        dock_height = 150

        self.dock.setGeometry(
            main_geom.right() + 10,
            main_geom.top(),
            dock_width,
            dock_height
        )

    def moveEvent(self, event):
        super().moveEvent(event)
        if self.dock.isFloating():
            self.position_dock()

    def toggle_dock(self):
        if self.dock.isVisible():
            self.dock.hide()
        else:
            self.dock.show()

    def update_toggle_button(self, visible):
        if visible:
            self.toggle_dock_action.setText("Hide Advanced Options")
        else:
            self.toggle_dock_action.setText("Show Advanced Options")

    def open_website(self):
        species = self.detection['Label'].replace("'", "").replace(" ", "_")
        url = f'https://www.allaboutbirds.org/guide/{species}/sounds'
        webbrowser.open(url)

    def arrow_activation(self):
        # Disable left arrow if at first detection
        self.left_arrow.setEnabled(self.counter > 0)
        # Disable right arrow if at last detection
        self.right_arrow.setEnabled(self.counter < len(self.species_detections) - 1)

    def check_behavior_criteria(self):
        label = self.detection['Label']
        behavior = self.behavior.get(label)

        if behavior is None:
            return  # No behavior set — skip logic

        if behavior == 'blacklist':
            self.output.at[self.period_counter, label] = 'Failed Verification'
            self.increment_species()

        elif behavior == 'whitelist':
            self.output.at[self.period_counter, label] = 'Confirmed Present'
            self.increment_species()

        elif behavior == 'graylist':
            # Ensure 'Date' is datetime
            if not pd.api.types.is_datetime64_any_dtype(self.output['Date']):
                self.output['Date'] = pd.to_datetime(self.output['Date'], errors='coerce')

            confirmed = self.output[self.output[label] == 'Confirmed Present']
            if confirmed.empty:
                return  # No previous confirmations, fall through

            last_date = confirmed['Date'].max()
            current_date = self.output.iloc[self.period_counter]['Date']

            # If dates are valid, check interval
            if pd.notnull(last_date) and pd.notnull(current_date):
                delta = (current_date - last_date).days
                if delta < 7:
                    self.output.at[self.period_counter, label] = 'Confirmed Present'
                    self.increment_species()

    def increment_species(self):
        self.species_counter = self.species_counter + 1
        progress = int(self.species_counter / len(self.species_list) * 100)
        self.progress.setValue(progress)
        self.counter = 0

        if self.species_counter == len(self.species_list):
            self.output.at[self.period_counter, 'Complete'] = 'Yes'
            self.initialize_period()
        else:
            self.species_detections = self.selection_df_final[
                self.selection_df_final['Label'] == self.species_list[self.species_counter]
            ].sort_values(by='Score', ascending=False)

            self.detection = self.species_detections.iloc[self.counter]

        self.check_behavior_criteria()
        species_text = f"{self.detection['Label']}"
        self.species_label.setText(species_text)

    def next_sound(self, new_species: bool, forward: bool = True, first_sound: bool = False) -> None:
        if first_sound:
            self.species_counter = -1
            self.increment_species()
            self.message.setText('Species Progress for Period:')

        self.disable_buttons()
        self.volume.setValue(20)
        checked_button = self.dock_buttons.checkedButton()
        self.behavior[self.detection['Label']] = checked_button.objectName()

        if not new_species:
            index_adjustment = 1 if forward else -1
            self.counter = self.counter + index_adjustment

        #Get specific detection
        self.detection = self.species_detections.iloc[self.counter]

        # Update info label
        info_text = f"Score: {self.detection['Score']} | Detection {self.counter + 1}/{self.species_detections.shape[0]}"
        self.info_label.setText(info_text)
        self.play_sound()

    def detection_decision(self, good_id: bool):
        if good_id:
            self.output.at[self.period_counter, self.detection['Label']] = 'Confirmed Present'
        else:
            self.output.at[self.period_counter, self.detection['Label']] = 'Failed Verification'
        self.increment_species()
        self.next_sound(new_species=True)

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

        # Update behavior button
        chosen = self.behavior.get(self.detection['Label'])
        if chosen == 'blacklist':
            self.black.setChecked(True)
        elif chosen == 'whitelist':
            self.white.setChecked(True)
        elif chosen == 'graylist':
            self.gray.setChecked(True)
        else:
            self.default.setChecked(True)

        start_time = self.detection['Begin Time (s)']
        end_time = self.detection['End Time (s)']
        y, self.sr = self.audio_files[self.detection['File']]
        self.part = y[int(start_time * self.sr):int(end_time * self.sr)]

        self.plot_spec()

        v = self.volume.value()/50

        self.worker = SoundThread(self.part, self.sr, v)
        self.worker.start()
        self.worker.finished.connect(self.activate_buttons)
        self.worker.finished.connect(self.arrow_activation)

    def export_current_clip(self):
        if self.part is None or self.detection is None:
            QMessageBox.warning(self, "Export Error", "No clip is loaded to export.")
            return

        species = self.detection['Label'].replace(" ", "_").replace("'", "")
        begin_seconds = self.detection['Begin Time (s)']

        # Get base recording time from the filename
        # Assumes format like: ID_20240612_054213.wav
        raw_filename = self.detection['File']
        date_time_match = re.search(r'(\d{8})_(\d{6})', raw_filename)

        if date_time_match:
            date_str = date_time_match.group(1)  # '20240612'
            time_str = date_time_match.group(2)  # '054213'
            dt = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
            obs_time = dt + timedelta(seconds=begin_seconds)
        else:
            QMessageBox.warning(self, "Export Error", "Could not parse datetime from filename.")
            return

        timestamp = obs_time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{species}.wav"
        export_dir = self.file_path / "Exported_Clips"
        export_dir.mkdir(exist_ok=True)

        filepath = export_dir / filename

        try:
            sf.write(filepath, self.part, self.sr)
            QMessageBox.information(self, "Export Successful", f"Clip exported to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not write file:\n{e}")

    def activate_buttons(self):
        self.goodID.setEnabled(True)
        self.badID.setEnabled(True)
        self.repeat.setEnabled(True)
        self.open_web.setEnabled(True)
        self.export_clip.setEnabled(True)

    def disable_buttons(self):
        self.goodID.setEnabled(False)
        self.badID.setEnabled(False)
        self.repeat.setEnabled(False)
        self.left_arrow.setEnabled(False)
        self.right_arrow.setEnabled(False)
        self.open_web.setEnabled(False)
        self.export_clip.setEnabled(False)

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

        self.output['Complete'] = 'No'
        self.output = self.output.drop(columns='PATH')
        self.initialize_period()

    def initialize_period(self):
        if self.period_counter == -1:
            if Path(self.file_path / 'results.csv').is_file():
                self.output = pd.read_csv(self.file_path / 'results.csv')
                self.period_counter = self.output.index[self.output['Complete'] == 'No'][0] - 1

            if Path(self.file_path / 'behavior_dict.csv').is_file():
                behavior_df = pd.read_csv(self.file_path / 'behavior_dict.csv', index_col=0)
                behavior = behavior_df.to_dict("split")
                self.behavior = dict(zip(behavior["index"], [row[0] for row in behavior["data"]]))

        # tick counter
        self.period_counter = self.period_counter + 1 # starts at 0

        # save output every period
        self.output.to_csv(self.file_path / 'results.csv', index=False)

        behavior_df = pd.DataFrame.from_dict(self.behavior, orient="index")
        behavior_df.reset_index(inplace=True)
        behavior_df.to_csv(self.file_path / 'behavior_dict.csv', index = False)

        if self.period_counter == self.output.shape[0]: # check if final period completed
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
            selection_df = DataFrame()
            for file in period_files['PATH']:
                selection_df= pd.read_csv(list(self.selection_path.glob(f'*{file}*'))[0], delimiter='\t')
                selection_df['File'] = file
                selection_df_list.append(selection_df)
                selection_df = pd.concat(selection_df_list, ignore_index=True)
            self.selection_df_final = []

            if 'Confidence' in selection_df.columns:
                selection_df.rename(columns={'Confidence':'Score'}, inplace=True)

            if 'Common Name' in selection_df.columns:
                selection_df.rename(columns={'Common Name':'Label'}, inplace=True)

            if 'Label' in selection_df.columns and not all(selection_df['Label'].str.islower()):
                for index, row in selection_df.iterrows():
                    if not row['Label'].islower(): # weird bug in Raven Pro where sometimes species names are abbreviated ex: baleag == Bald Eagle
                        self.selection_df_final.append(row)
                self.selection_df_final = pd.DataFrame(self.selection_df_final)
                self.species_list = self.selection_df_final.loc[:, 'Label'].unique()

                for spec in self.species_list:
                    if 'spec' in self.output.columns:
                        continue
                    else: self.output[spec] = ''

                total_files = len(period_files['PATH'])
                self.audio_files = {}
                self.message.setText('Loading Audio Files...')
                for index, file in enumerate(period_files['PATH']):
                    wav_path = list(self.data_path.glob(f'*{file}*'))
                    self.audio_files[file] = librosa.load(wav_path[0])
                    progress = int((index + 1) / total_files * 100)
                    self.progress.setValue(progress)
                self.next_sound(new_species=True, first_sound=True, forward=True)

            else:
                self.output.at[self.period_counter, 'Complete'] = 'Yes'
                self.output.drop(columns='PATH').to_csv(self.file_path / 'results.csv', index=False)
                self.initialize_period()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = AudioAnalysis()
    main.start()
    sys.exit(app.exec_())