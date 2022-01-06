from sklearn.model_selection import KFold, train_test_split
import argparse
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments for patches creation.')
    parser.add_argument("--data_path", type=str, required = True,
                        help='path to folder containing train/val/test folders')
    args = parser.parse_args()
    subjects = []
    for data_type in ["train", "val", "test"]:
        subjects += os.listdir("%s/%s" % (args.data_path, data_type))
    subjects = np.asarray(subjects)
    kf = KFold(n_splits = 10)
    cpt = 0
    for train, test in kf.split(subjects):
        #we get train / test indexes
        train, val = train_test_split(train, test_size = len(test))
        f = open("%s/cv_group_%s.txt" % (args.data_path, cpt), "w")
        # f.writelines(["%s " % sub for sub in subjects[np.asarray(train)]])
        # f.writelines("\n")
        f.writelines(["%s " % sub for sub in subjects[np.asarray(val)]])
        f.writelines("\n")
        f.writelines(["%s " % sub for sub in subjects[np.asarray(test)]])
        f.writelines("\n")
        f.close()
        cpt += 1
    print("KFold CV groups done")
