from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import torch
import towards_biden
import towards_trump

def save(biden_result, trump_result):
    # TODO save the result
    return

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Add models\' arguments')
    parser.add_argument('--tw', default='both', 
                    help='towards which person, t->trump, b->biden, default->both')
    args = parser.parse_args()

    paths = []

    if args.type == 'b':
        towards_biden.predict(paths, device)
    elif args.type == 't':
        towards_trump.predict(paths, device)
    elif args.type == 'both':
        biden_result = towards_biden.predict(paths, device)
        trump_result = towards_trump.predict(paths, device)
        save(biden_result, trump_result)