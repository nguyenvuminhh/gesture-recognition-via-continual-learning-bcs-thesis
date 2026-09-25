"""
Evaluate the final checkpoint of each experiment on the test split (Final Test Accuracy)
and write the results to test_results.json.

Expects checkpoints at log/classification/final_<experiment>/checkpoints/final_model.pth.
"""
import argparse
import importlib
import json
import os
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from data_utils.ModelNetDataset import ModelNetDataset
from train_classification import DATA_PATH, evaluate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

EXPERIMENTS = ['joint_224', 'none', 'ewc', 'si', 'lwf', 'er', 'agem']


def parse_args():
    parser = argparse.ArgumentParser('testing')
    parser.add_argument('--experiments', nargs='+', default=EXPERIMENTS, help='experiment names to evaluate')
    parser.add_argument('--model', default='pointnet2-bi-gru', help='model module in models/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_category', type=int, default=16)
    parser.add_argument('--num_point', type=int, default=3360)
    parser.add_argument('--use_KAN', action='store_true')
    parser.add_argument('--output', default='test_results.json')
    return parser.parse_args()


def main(args):
    torch.cuda.empty_cache()

    # Dataset options matching the training defaults
    dataset_args = SimpleNamespace(
        num_point=args.num_point,
        num_category=args.num_category,
        use_normals=True,
        use_uniform_sample=False,
        continual_learning=True,
    )
    test_dataset = ModelNetDataset(root=DATA_PATH, args=dataset_args, split='test', process_data=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_module = importlib.import_module(args.model)

    results = {}
    for name in args.experiments:
        checkpoint = torch.load(f'log/classification/final_{name}/checkpoints/final_model.pth', map_location='cpu')
        classifier = model_module.GRUPointNet(args.num_category, normal_channel=True, use_KAN=args.use_KAN)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier = classifier.cuda()

        with torch.no_grad():
            _, accuracy = evaluate(classifier, test_loader, use_cpu=False)
        print(f'Experiment: {name}, Accuracy: {accuracy}')
        results[name] = accuracy

    with open(args.output, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main(parse_args())
