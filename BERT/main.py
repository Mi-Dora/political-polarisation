from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import torch
import pred_save
import os

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Add models\' arguments')
    parser.add_argument('--paths', default='all', help='paths')
    parser.add_argument('--url', default='../data_cleaned/tweets/', help='base url')
    args = parser.parse_args()
    files = []
    base_url = args.url
    if args.paths == 'all':
        for root,dirs,fs in os.walk(base_url):
            files = fs
    else :
        paths = args.paths.split(',')
        files = [base_url+'out_'+path+'pkl' for path in paths]

    pred_save.predict(files, device)