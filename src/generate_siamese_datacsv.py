import os
import argparse
import pandas as pd
import random
from itertools import combinations

random.seed(1)


def main(cropped):
    if cropped:
        dirt = 'cropped/PlantDoc-Dataset/'
    else:
        dirt = 'PlantDoc-Dataset/'
    for i in ['train', 'val']:
        df1 = pd.DataFrame(columns=['filename1', 'filename2', 'class'])
        for subdir, dirs, files in os.walk(dirt + i):
            print("here1")
            filepath = [os.path.join(subdir, file) for file in files if os.path.join(subdir, file).endswith('.jpg')]
            if files:
                for file1, file2 in combinations(filepath, 2):
                    df1 = df1.append({'filename1': file1, 'filename2': file2, 'class': 1}, ignore_index=True)
        print("here")
        df0 = pd.DataFrame(columns=['filename1', 'filename2', 'class'])
        l = len(df1.index)
        for index, row in df1.iterrows():
            while True:
                r = random.randint(0, l - 1)
                if row['filename2'].split('/')[3] != df1.iloc[r]['filename2'].split('/')[3]:
                    break
            print(index)
            df0 = df0.append({'filename1': row['filename1'], 'filename2': df1.iloc[r]['filename2'], 'class': 0}, ignore_index=True)

        if cropped:
            ((pd.concat([df0, df1], ignore_index=True)).sample(frac=1).reset_index(drop=True)).to_csv(i + '_siamese_data_cropped.csv', index=False)
        else:
            ((pd.concat([df0, df1], ignore_index=True)).sample(frac=1).reset_index(drop=True)).to_csv(i + '_siamese_data.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cropped", action='store_true',
                        help="whether to generate training data csv for Siamese network on cropped dataset")
    args = parser.parse_args()
    main(args.cropped)
