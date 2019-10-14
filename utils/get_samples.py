from pathlib import Path
import os
import requests
import zipfile

current_directory = Path(os.path.abspath(__file__)).parent
sample_directory = current_directory.joinpath('APTMalware', 'samples')

failed = []
succeeded = []
samples_to_query = []


def perform_submit_request(filename):
    input_zip = zipfile.ZipFile(str(filename))
    input_zip.setpassword(b"infected")
    samplename = input_zip.namelist()[0]
    #data = input_zip.read(samplename)
    #rel_path = str(Path(filename.replace('.zip', '')).relative_to(Path(filename).parent.parent))
    print(samplename)
    #succeeded.append((rel_path, task_id))
    #failed.append(filename)


for (dirpath, dirnames, filenames) in os.walk(str(sample_directory)):
    samples_to_query.extend([str(Path(dirpath).joinpath(f)) for f in filenames])

i = 1

for sample in samples_to_query:
    perform_submit_request(sample)
    print(i, sample)
    i += 1

with open(str(current_directory.joinpath('failed.txt')), 'w') as f:
    f.write('\n'.join(failed))

with open(str(current_directory.joinpath('succeeded.txt')), 'w') as f:
    f.write('\n'.join('%s - %s' % x for x in succeeded))
