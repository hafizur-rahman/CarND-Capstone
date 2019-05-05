"""
Usage:

# Data downloaded from:
# https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view
# https://1drv.ms/u/s!AtMG4jW974a6m8B-Q0A3tc2JbCigPw

# Create train/test data:

python preprocess.py -i [PATH_TO_ANNOTATIONS_FILE] \
    -o [PATH_TO_ANNOTATIONS_FOLDER] \
    -r 0.2
"""

import os
import glob
import pandas as pd
import argparse
import yaml
import sys

from sklearn.model_selection import train_test_split

def txt2df(inputFile):
    class_ids = [
        'Red', 'Yellow','Green'
    ]

    with open(inputFile, "r+") as f:
        items = []

        for line in f:
            # 'image-file-basename classno 1 boundingbox-xmin ymin width height'
            # 0/1521230383.77.png 0 2 35 99 84 170 393 103 78 175
            data = line.split()

            file_name=data[0]
            class_id=class_ids[int(data[1])]
            bboxes=int(data[2])

            for i in range(bboxes):
                xmin = int(data[i*4 + 3])
                ymin = int(data[i*4 + 4])
                width = int(data[i*4 + 5])
                height = int(data[i*4 + 6])
                xmax = xmin + width
                ymax = ymin + height

                items.append((file_name, width, height, class_id, xmin, ymin, xmax, ymax))            

        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        df = pd.DataFrame(items, columns=column_name)
    return df

def yml2df(inputFile):
    with open(inputFile, "r+") as f:
        data = yaml.safe_load(f)

        items = []

        for entry in data:
            filename = entry['filename']

            for a in entry['annotations']:
                items.append((filename, a['x_width'], a['y_height'], a['class'], a['xmin'], 
                            a['ymin'], a['xmin'] + a['x_width'], a['ymin'] + a['y_height']))

        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        df = pd.DataFrame(items, columns=column_name)
        
        return df


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Annonation to-CSV converter")
    parser.add_argument("-i",
                        "--inputFile",
                        help="Path to the annotation file",
                        type=str)
    parser.add_argument("-o",
                        "--outputPath",
                        help="Name of csv file output dir",
                        type=str)
    parser.add_argument("-r",
                        "--ratio",
                        help="train/test split ration",
                        type=float,
                        default=0.0)
    args = parser.parse_args()

    assert(os.path.isfile(args.inputFile))
    assert(os.path.isdir(args.outputPath))

    if args.inputFile.endswith(".txt"):
        df = txt2df(args.inputFile)
    elif args.inputFile.endswith(".yml"):
        df = yml2df(args.inputFile)
    else:
        print('Unrecognized format: {}'.format(args.inputFile))
        sys.exit(-1)

    # Shuffle Pandas data frame
    import sklearn.utils
    df = sklearn.utils.shuffle(df)

    train, test = train_test_split(df, test_size=args.ratio)

    train_labels = os.path.join(args.outputPath, 'train_labels.csv')
    train.to_csv(train_labels, index=None)
    print('Train labels saved at: {}'.format(train_labels))

    if len(test):
        test_labels = os.path.join(args.outputPath, 'test_labels.csv')
        test.to_csv(test_labels, index=None)
        print('Test labels saved at: {}'.format(test_labels))

if __name__ == '__main__':
    main()