from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import torch
import towards_biden
import towards_trump
import os

def save(biden_result, trump_result):
    # TODO save the result
    return

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Add models\' arguments')
    parser.add_argument('--tw', default='both', 
                    help='towards which person, t->trump, b->biden, default->both')
    parser.add_argument('--paths', default='all', help='paths')
    parser.add_argument('--url', default='../data_cleaned/tweets/', help='base url')
    args = parser.parse_args()

    base_url = args.url
    if args.paths == 'all':
        for root,dirs,files in os.walk(base_url):
            paths = files
    else :
        paths = args.paths.split(',')
        paths = [base_url+'out_'+path+'pkl' for path in paths]

    if args.type == 'b':
        towards_biden.predict(paths, device)
    elif args.type == 't':
        towards_trump.predict(paths, device)
    elif args.type == 'both':
        biden_result = towards_biden.predict(paths, device)
        trump_result = towards_trump.predict(paths, device)
        save(biden_result, trump_result)