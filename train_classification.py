"""
Continual learning training loop for the PointNet++ / GRU gesture classifier.

Trains one method (NONE, JOINT, EWC, SI, LwF, ER, AGEM) over all contexts in
DATA_SET_NAME_LIST, evaluating current-context and seen-context accuracy after
each context. Skeleton adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch.
"""
import argparse
import datetime
import importlib
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

import provider
from data_utils.constants import DATA_SET_NAME_LIST, NOF_CONTEXTS, ContinualLearningMethodClass
from data_utils.ModelNetDataset import ModelNetDataset
from models.memory_buffer import MemoryBuffer
from plotting_scripts.result_dto import DotProductDTO, TrainingResultDTO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

DATA_PATH = './data/bigdata_12_classes/'
LOG_ROOT = Path('./log/classification')

# Method hyperparameters that are not exposed on the command line
NOF_ACTIVE_CLASSES = 12  # classes present in the domain-incremental contexts
LWF_TEMPERATURE = 2.0
AGEM_GAMMA = 1e-7
REPLAY_BATCH_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--method', type=ContinualLearningMethodClass, help='continual learning method')
    parser.add_argument('--reg_strength', type=float, default=1e-4, help='lambda for EWC / SI / LwF')
    parser.add_argument('--buffer_size', type=float, default=0, help='replay buffer budget for ER / AGEM')
    parser.add_argument('--use_cpu', action='store_true', help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='pointnet2-bi-gru', help='model module in models/')
    parser.add_argument('--num_category', type=int, default=16, help='total number of classes')
    parser.add_argument('--epoch', type=int, default=224, help='total epochs, split evenly across contexts')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num_point', type=int, default=3360, help='points per sample (28 frames x 120 points)')
    parser.add_argument('--use_normals', action='store_true', default=True, help='use extra point features')
    parser.add_argument('--process_data', action='store_true', help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', help='use uniform sampling')
    parser.add_argument('--use_KAN', action='store_true', help='use KAN instead of MLP classifier')
    parser.add_argument('--starting_model_path', type=str, default=None, help='path to starting model checkpoint')
    parser.add_argument('--starting_context_index', type=int, default=0, help='index of starting context')
    parser.add_argument('--dry_run', action='store_true', help='one epoch per context, for testing')
    return parser.parse_args()


def inplace_relu(m):
    if 'ReLU' in m.__class__.__name__:
        m.inplace = True


def evaluate(model, loader, use_cpu):
    """Return (macro F1, accuracy) of `model` on `loader`."""
    y_true, y_pred = [], []
    model.eval()
    for points, target in tqdm(loader, total=len(loader)):
        if not use_cpu:
            points, target = points.cuda(), target.cuda()
        pred = model(points.transpose(2, 1))
        y_true.extend(target.long().cpu().numpy())
        y_pred.extend(pred.data.max(1)[1].long().cpu().numpy())

    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    return f1, accuracy


def make_loader(dataset, batch_size, train=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0, drop_last=train)


def build_classifier(model_module, args):
    classifier = model_module.GRUPointNet(args.num_category, normal_channel=args.use_normals, use_KAN=args.use_KAN)
    if args.starting_model_path:
        checkpoint = torch.load(args.starting_model_path, map_location='cpu')
        classifier.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    classifier.apply(inplace_relu)
    return classifier if args.use_cpu else classifier.cuda()


def build_criterion(model_module, classifier, train_loader, args):
    if args.method == ContinualLearningMethodClass.EWC:
        criterion = model_module.get_ewc_loss(
            model=classifier,
            ewc_lambda=args.reg_strength,
            dataloader=train_loader,
            nof_active_classes=NOF_ACTIVE_CLASSES,
            args=args,
        )
    elif args.method == ContinualLearningMethodClass.SI:
        criterion = model_module.get_si_loss(model=classifier, si_lambda=args.reg_strength)
    elif args.method == ContinualLearningMethodClass.LwF:
        criterion = model_module.get_lwf_loss(
            model=classifier, lwf_lambda=args.reg_strength, temperature=LWF_TEMPERATURE,
        )
    else:
        criterion = model_module.get_loss()
    return criterion if args.use_cpu else criterion.cuda()


def agem_project(classifier, model_module, criterion, memory_buffer):
    """Replace the current gradients with the A-GEM projected gradient.

    Returns a DotProductDTO describing how much the projection changed the gradient.
    """
    replay_dataset = memory_buffer.get_dataset(batch_size=REPLAY_BATCH_SIZE)
    g_current = model_module.get_current_gradients(classifier)
    g_replay = model_module.get_replay_gradients(classifier, replay_dataset, provider.data_augmentation, criterion)
    dot_product = torch.dot(g_current, g_replay)

    gradient = (
        g_current
        if dot_product >= 0 else
        g_current + (dot_product * g_replay) / (dot_product + AGEM_GAMMA)
    )

    eps = 1e-12
    stats = DotProductDTO(
        cos_sim=(torch.dot(g_current, gradient) / (g_current.norm() * gradient.norm() + eps)).item(),
        magnitude_ratio=((gradient.norm() + eps) / (g_current.norm() + eps)).item(),
        projection_loss=torch.norm(g_current - gradient).item(),
    )

    idx = 0
    for p in classifier.parameters():
        if p.grad is not None:
            numel = p.numel()
            p.grad.copy_(gradient[idx:idx + numel].view_as(p))
            idx += numel
    return stats


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.empty_cache()

    continual_learning = args.method is not ContinualLearningMethodClass.JOINT
    need_memory_buffer = args.method in (ContinualLearningMethodClass.ER, ContinualLearningMethodClass.AGEM)
    args.continual_learning = continual_learning  # read by ModelNetDataset

    nof_epoch_per_context = args.epoch // NOF_CONTEXTS
    args.epoch = nof_epoch_per_context * NOF_CONTEXTS
    if args.dry_run:
        args.epoch = len(DATA_SET_NAME_LIST)
    elif len(DATA_SET_NAME_LIST) != 16:
        raise ValueError(f'Expected 16 contexts in DATA_SET_NAME_LIST, found {len(DATA_SET_NAME_LIST)}.')

    '''CREATE DIR'''
    now = datetime.datetime.now()
    run_name = (
        f"date_{now.strftime('%B').lower()}_{now.strftime('%d')}_"
        f"time_{now.strftime('%H%M')}_"
        f"method_{args.method}_"
        f"lambda_{args.reg_strength}_"
        f"buffer_{args.buffer_size}_"
    )
    exp_dir = LOG_ROOT / run_name
    checkpoints_dir = exp_dir / 'checkpoints'
    log_dir = exp_dir / 'logs'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger('Model')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_dir / f'{args.model}.txt')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    def log_string(msg):
        logger.info(msg)
        print(msg)

    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    train_dataset = ModelNetDataset(root=DATA_PATH, args=args, split='train', process_data=args.process_data)
    val_dataset = ModelNetDataset(root=DATA_PATH, args=args, split='val', process_data=args.process_data)
    val_so_far_dataset = ModelNetDataset(root=DATA_PATH, args=args, split='val_so_far', process_data=args.process_data)

    train_loader = make_loader(train_dataset, args.batch_size, train=True)
    val_loader = make_loader(val_dataset, args.batch_size)
    val_so_far_loader = make_loader(val_so_far_dataset, args.batch_size)

    '''MODEL LOADING'''
    model_module = importlib.import_module(args.model)
    # Snapshot the code used for this run next to its results
    shutil.copy(f'./models/{args.model}.py', exp_dir)
    shutil.copy('./models/pointnet2_utils.py', exp_dir)
    shutil.copy('./train_classification.py', exp_dir)

    classifier = build_classifier(model_module, args)
    criterion = build_criterion(model_module, classifier, train_loader, args)

    nof_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    log_string(f'Total number of parameters: {nof_params}')
    log_string(f'There are {NOF_CONTEXTS} contexts in total.')
    log_string(f'Total epochs: {args.epoch}')
    log_string(f'Epochs per context: {nof_epoch_per_context}')

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    memory_buffer = None
    if need_memory_buffer:
        memory_buffer = MemoryBuffer(budget=args.buffer_size)
        log_string(f'Using memory buffer with {args.buffer_size} samples')

    '''TRAINING'''
    train_losses, train_accuracies = [], []
    context_f1s, context_accuracies = [], []
    so_far_f1s, so_far_accuracies = [], []
    dot_products: dict[int, list[DotProductDTO]] = {}
    # The final test accuracy is computed separately by test_models.py
    final_test_f1 = final_test_accuracy = 0

    def validate():
        context_f1, context_accuracy = evaluate(classifier, val_loader, args.use_cpu)
        so_far_f1, so_far_accuracy = evaluate(classifier, val_so_far_loader, args.use_cpu)
        context_f1s.append(context_f1)
        context_accuracies.append(context_accuracy)
        so_far_f1s.append(so_far_f1)
        so_far_accuracies.append(so_far_accuracy)
        log_string('----------------')
        log_string(f'[Context] Val F1 Score: {context_f1:f}')
        log_string(f'[So Far] Val F1 Score: {so_far_f1:f}')
        log_string(f'[Context] Val Accuracy: {context_accuracy:f}')
        log_string(f'[So Far] Val Accuracy: {so_far_accuracy:f}')
        log_string('----------------')

    global_epoch = args.starting_context_index * nof_epoch_per_context
    start_time = time.time()
    log_string('Start training...')

    while global_epoch < args.epoch:
        torch.cuda.empty_cache()
        log_string(f'Epoch {global_epoch + 1}/{args.epoch}:')
        classifier.train()
        scheduler.step()

        if args.method == ContinualLearningMethodClass.ER and memory_buffer and len(memory_buffer) > 0:
            train_loader = make_loader(
                ConcatDataset([train_dataset, memory_buffer.get_dataset()]), args.batch_size, train=True,
            )

        mean_correct = []
        for points, target in tqdm(train_loader, total=len(train_loader), smoothing=0.9):
            classifier.train()
            optimizer.zero_grad()

            points = provider.data_augmentation(points)
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred = classifier(points)
            loss = criterion(pred, target.long(), inputs=points)
            correct = pred.data.max(1)[1].eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()

            if args.method == ContinualLearningMethodClass.AGEM and memory_buffer and len(memory_buffer) > 0:
                stats = agem_project(classifier, model_module, criterion, memory_buffer)
                dot_products.setdefault((global_epoch + 1) // nof_epoch_per_context, []).append(stats)

            optimizer.step()
            criterion.run_post_iteration()

        train_accuracy = sum(mean_correct) / len(mean_correct)
        train_losses.append(loss.item())
        train_accuracies.append(train_accuracy)
        log_string(f'Train loss: {loss.item():f}')
        log_string(f'Train accuracy: {train_accuracy:f}')

        global_epoch += 1

        if not (continual_learning and global_epoch % nof_epoch_per_context == 0):
            continue

        '''END OF CONTEXT'''
        with torch.no_grad():
            validate()

        context_index = global_epoch // nof_epoch_per_context - 1
        context_name = DATA_SET_NAME_LIST[context_index] if context_index < len(DATA_SET_NAME_LIST) else 'Unknown'
        log_string(f'context {context_index + 1} ({context_name}) finished, start next context')

        if need_memory_buffer:
            log_string('Adding new context to memory buffer...')
            memory_buffer.add_data(train_dataset)
        criterion.run_post_context()  # EWC: Fisher, SI: omega, LwF: snapshot old model

        state = {
            'epoch': global_epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
        }
        torch.save(state, checkpoints_dir / f'context_{context_index + 1}_model.pth')
        torch.save(state, checkpoints_dir / 'last_model.pth')
        log_string(f'Saved checkpoint for context {context_index + 1}')

        train_dataset = train_dataset.next_context()
        val_dataset = val_dataset.next_context()
        val_so_far_dataset = val_so_far_dataset.next_context()
        if train_dataset is None or val_dataset is None or val_so_far_dataset is None:
            log_string('No more contexts available, stopping training.')
            break

        train_loader = make_loader(train_dataset, args.batch_size, train=True)
        val_loader = make_loader(val_dataset, args.batch_size)
        val_so_far_loader = make_loader(val_so_far_dataset, args.batch_size)
        if args.method == ContinualLearningMethodClass.EWC:
            criterion.set_dataloader(train_loader)

    if not continual_learning:
        with torch.no_grad():
            validate()

    training_time = time.time() - start_time
    log_string(f'Total training time: {training_time / 60:.2f} minutes, '
               f'per epoch: {training_time / 60 / global_epoch:.2f} minutes')

    '''SAVE MODEL AND RESULTS'''
    torch.save({'model_state_dict': classifier.state_dict()}, checkpoints_dir / 'final_model.pth')

    training_result = TrainingResultDTO(
        method_used=args.method,
        nof_context=NOF_CONTEXTS,
        datasets=DATA_SET_NAME_LIST,
        number_of_epochs=args.epoch,
        training_time=training_time,
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        context_f1s=context_f1s,
        so_far_f1s=so_far_f1s,
        context_accuracies=context_accuracies,
        so_far_accuracies=so_far_accuracies,
        final_test_accuracy=final_test_accuracy,
        final_test_f1=final_test_f1,
        dot_products=dot_products,
    )
    result_path = exp_dir / 'training_result.json'
    result_path.write_text(training_result.model_dump_json(indent=4))
    log_string(f'Training result saved to {result_path}')


if __name__ == '__main__':
    main(parse_args())
