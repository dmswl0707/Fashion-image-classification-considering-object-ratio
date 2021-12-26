import glob
import piexif
from imgnet_preprocessing.imagennet_class_index import train_path, val_path

n_files = 0
for filename in glob.iglob(train_path, recursive=True):
    n_files += 1
    print("About to process file %d, which is %s." % (n_files,filename))
    piexif.remove(filename)

