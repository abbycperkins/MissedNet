from birdnetlib.batch import DirectoryMultiProcessingAnalyzer
from birdnetlib.analyzer import Analyzer
from datetime import datetime
from pprint import pprint

latitude = 37.088357
longitude = -95.909164
directory = 'D:/ARU_2024/HavanaLake/Data'

def on_analyze_directory_complete(recordings):
    print("-" * 80)
    print("directory_completed: recordings processed ", len(recordings))
    print("-" * 80)

    for recording in recordings:
        print(recording.path)
        if recording.error:
            print("Error: ", recording.error_message)
        else:
            pprint(recording.detections)

        print("-" * 80)


def preanalyze(recording):
    # Used to modify the recording object before analyzing.
    filename = recording.filename
    dt = datetime.strptime(filename.as_posix().split(sep='_')[-2],'%Y%m%d')
    # Modify the recording object here as needed.
    recording.date = dt

analyzer = Analyzer()

directory = "."
batch = DirectoryMultiProcessingAnalyzer(
    directory,
    analyzers=[analyzer],
    lon=longitude,
    lat=latitude,
    min_conf=0.25,
)

batch.recording_preanalyze(preanalyze)
batch.on_analyze_directory_complete = on_analyze_directory_complete
batch.process()