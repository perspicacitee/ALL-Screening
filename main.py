import json
import os
import random
import time
import cv2
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from easydict import EasyDict as Edict
from models import ResNet, Acute
from models.model import ImageModel, TabularModel
from utils import EarlyStopping, save_roc_curve, metrics
from utils.scheduler import WarmupCosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from data.Dataset import TotalDataset
# from data.Dataset import ClnDataset as NewDataset
import yaml
from multiprocessing import Process, Manager, Pool
import logging
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_logger(LEVEL, log_file=None):
    head = '[%(asctime)-15s] [%(levelname)s] %(message)s'
    if LEVEL == 'info':
        logging.basicConfig(level=logging.INFO, format=head)
    elif LEVEL == 'debug':
        logging.basicConfig(level=logging.DEBUG, format=head)
    logger = logging.getLogger()
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
    return logger


@torch.no_grad()
def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.normal_(m.bias)


@torch.no_grad()
def evaluate(test_dataloader, model, criterion, args):
    preds = []
    labels = []
    losses = []

    model.eval()
    for idx, data in enumerate(test_dataloader):
        img = data['image'].to(args.device)
        cln_data = data["血常规报告参数"]
        cln_data = torch.stack(list(cln_data.values()), dim=1).to(args.device, dtype=torch.float)
        # if args.mask is not None:
            # cln_data[:, args.mask] = 0
        label = data['label'].to(args.device)
        logit = model(img, cln_data)
        # logit = model(cln_data)

        pred = F.softmax(logit)
        # pred = model(img)
        loss = criterion(logit, label)
        groud_truth = torch.zeros_like(pred, device=args.device).scatter_(1, label.to(torch.int64).unsqueeze(1), 1)
        # loss = criterion(pred, groud_truth)
        preds.append(pred.detach().cpu())
        labels.append(groud_truth.detach().cpu())
        losses.append(loss.item())

    y_score = torch.cat(preds, dim=0)
    y_label = torch.cat(labels, dim=0)
    loss = np.mean(losses)

    return loss, y_score, y_label


def train(model, train_dataloader, test_dataloader, criterion, optim, num_epochs, args: Edict):
    train_loss = []
    train_epochs_loss = []
    train_epochs_acc = []
    valid_epochs_acc = []
    evaluate_metrics = {}
    best_acc = np.NINF

    early_stopping = EarlyStopping(path=os.path.join("./checkpoint/", args.exp_name), patience=50,
                                   verbose=True, ascending=False)
    writter = SummaryWriter('./result/log/' + args.exp_name)
    logger = get_logger('info', os.path.join('./result/log/', args.exp_name + '.log'))

    in_sample = torch.randn(args.batch_size, 3, 224, 224).to(args.device)  # B, C, H, W
    cln_sample = torch.rand(args.batch_size, 25).to(args.device)
    writter.add_graph(model, (in_sample, cln_sample))

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.num_epochs,
    #                                                        eta_min=1e-7, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=int(0.3 * args.epochs), gamma=0.1)
    # warm up
    scheduler = WarmupCosineAnnealingLR(optim, warmup_epochs=10, max_epochs=args.num_epochs, eta_min=1e-9)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda epoch: 1e-4 * (epoch + 1) / 10)
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_batch_loss = []
        train_batch_acc = []

        for idx, data in enumerate(train_dataloader):
            img = data['image'].to(args.device)
            cln_data = data["血常规报告参数"]
            cln_data = torch.stack(list(cln_data.values()), dim=1).to(args.device, dtype=torch.float32)
            # if args.mask is not None:
            #     cln_data[:, args.mask] = 0
            label = data['label'].to(args.device)

            logit = model(img, cln_data)
            pred = F.softmax(logit)
            loss = criterion(logit, label)
            # pred = model(img)
            groud_truth = torch.zeros_like(pred).scatter_(1, label.to(torch.int64).unsqueeze(1), 1)
            # loss = criterion(pred, groud_truth)

            optim.zero_grad()
            loss.backward()
            optim.step()

            acc = torch.sum(pred * groud_truth >= 0.5) / torch.sum(groud_truth)
            train_batch_loss.append(loss.item())
            train_loss.append(loss.item())
            train_batch_acc.append(acc.item())
            if idx % 10 == 0:
                logger.info("epoch={}/{},batch={}/{}of train, loss={}".format(
                    epoch, args.num_epochs, idx + 1, len(train_dataloader), loss.item()))

        # ==== save epoch loss ====
        train_epochs_loss.append(np.mean(train_batch_loss))
        train_epochs_acc.append(np.mean(train_batch_acc))
        writter.add_scalar("Train/Loss", train_epochs_loss[-1], epoch)
        writter.add_scalar("Train/Acc", train_epochs_acc[-1], epoch)

        # ==== valid ====
        logger.info("epoch={}/{}, start valid".format(epoch, args.num_epochs))
        loss, y_score, y_label = evaluate(test_dataloader, model, criterion, args)
        logger.info("epoch={}/{}, stop valid".format(epoch, args.num_epochs))
        y_pred = torch.argmax(y_score, dim=1)
        y_truth = torch.argmax(y_label, dim=1)
        acc_total = torch.sum(y_pred == y_truth).item() / len(y_truth)
        writter.add_scalar("Valid/Acc/total", acc_total, epoch)

        # region save tensorboard
        writter.add_scalar("Valid/loss", loss, epoch)
        # hetero Lmyp disease
        y_score_malignant, y_malignant = y_score[:, 2].cpu().numpy(), y_label[:, 2].cpu().numpy()
        auc_hetero = save_roc_curve(writter, y_malignant, y_score_malignant, epoch, tag="Valid/ROC_curve/hetero", exp_name=args.exp_name)
        acc_hetero, prec_hetero, rec_hetero, f1_score_hetero = metrics(y_pred.cpu().numpy() == 2, y_malignant)
        writter.add_scalar("Valid/Acc/hetero", acc_hetero, epoch)
        writter.add_scalar("Valid/Auc/hetero", auc_hetero, epoch)
        writter.add_scalar("Valid/Precision/hetero", prec_hetero, epoch)
        writter.add_scalar("Valid/Recall/hetero", rec_hetero, epoch)
        writter.add_scalar("Valid/F1_score/hetero", f1_score_hetero, epoch)

        # acute Lmyp disease
        y_score_malignant, y_malignant = y_score[:, 0].cpu().numpy(), y_label[:, 0].cpu().numpy()
        auc_acute = save_roc_curve(writter, y_malignant, y_score_malignant, epoch, tag="Valid/ROC_curve/acute", exp_name=args.exp_name)
        acc_acute, prec_acute, rec_acute, f1_score_acute = metrics(y_pred.cpu().numpy()==0, y_malignant)
        writter.add_scalar("Valid/Acc/acute", acc_acute, epoch)
        writter.add_scalar("Valid/Auc/acute", auc_acute, epoch)
        writter.add_scalar("Valid/Precision/acute", prec_acute, epoch)
        writter.add_scalar("Valid/Recall/acute", rec_acute, epoch)
        writter.add_scalar("Valid/F1_score/acute", f1_score_acute, epoch)

        # normal Lmyp disease
        y_score_malignant, y_malignant = y_score[:, 1].cpu().numpy(), y_label[:, 1].cpu().numpy()
        auc_normal = save_roc_curve(writter, y_malignant, y_score_malignant, epoch, tag="Valid/ROC_curve/normal", exp_name=args.exp_name)
        acc_normal, prec_normal, rec_normal, f1_score_normal = metrics(y_pred.cpu().numpy()==1, y_malignant)
        writter.add_scalar("Valid/Acc/normal", acc_normal, epoch)
        writter.add_scalar("Valid/Auc/normal", auc_normal, epoch)
        writter.add_scalar("Valid/Precision/normal", prec_normal, epoch)
        writter.add_scalar("Valid/Recall/normal", rec_normal, epoch)
        writter.add_scalar("Valid/F1_score/normal", f1_score_normal, epoch)
        # if epoch % 10 == 1:
        writter.add_pr_curve(f"PR-Curve", y_malignant, y_score_malignant, epoch)

        # endregion
        valid_epochs_acc.append(acc_total)

        # region early stop
        if epoch % 10 == 1:  # save model parameters
            torch.save(model.state_dict(),
                       './checkpoint/' + args.exp_name + f"/model_checkpoint_epoch{epoch}.pth")
        early_stopping(loss, model)
        if early_stopping.early_stop:
            # break
            pass
        # endregion

        scheduler.step()
        writter.add_scalar("Train/lr", scheduler.get_last_lr()[0], epoch)

        # ==== save the data ====
        if not os.path.exists(os.path.join(os.getcwd(), 'result/runs', args.exp_name)):
            os.mkdir(os.path.join(os.getcwd(), 'result/runs', args.exp_name))
        if epoch % 10 == 0:
            save_path = os.path.join(os.getcwd(), 'result/runs', args.exp_name, f"epoch{epoch}.txt")
            np.savetxt(save_path,
                       np.concatenate((y_score.transpose(0, 1).cpu(), y_label.transpose(0, 1).cpu()), axis=0),
                       delimiter=',',
                       fmt="%.4f")
        if acc_total > best_acc:
            best_acc = acc_total

            # region Save the best result
            evaluate_metrics["epoch"] = epoch
            evaluate_metrics["acc"] = acc_total
            evaluate_metrics["acc_hetero"] = acc_hetero
            evaluate_metrics["acc_acute"] = acc_acute
            evaluate_metrics["acc_normal"] = acc_normal
            evaluate_metrics["auc_hetero"] = auc_hetero
            evaluate_metrics["auc_acute"] = auc_acute
            evaluate_metrics["auc_normal"] = auc_normal
            evaluate_metrics["prec_hetero"] = prec_hetero
            evaluate_metrics["prec_acute"] = prec_acute
            evaluate_metrics["prec_normal"] = prec_normal
            evaluate_metrics["rec_hetero"] = rec_hetero
            evaluate_metrics["rec_acute"] = rec_acute
            evaluate_metrics["rec_normal"] = rec_normal
            evaluate_metrics["f1_score_hetero"] = f1_score_hetero
            evaluate_metrics["f1_score_acute"] = f1_score_acute
            evaluate_metrics["f1_score_normal"] = f1_score_normal
            # endregion

            save_path = os.path.join(os.getcwd(), 'result/runs', args.exp_name, f"best_epoch.txt")
            np.savetxt(save_path,
                       np.concatenate((y_score.transpose(0, 1).cpu(), y_label.transpose(0, 1).cpu()), axis=0),
                       delimiter=',',
                       fmt="%.4f")
            torch.save(model.state_dict(),
                       './checkpoint/' + args.exp_name + f"/model_checkpoint_epoch{epoch}.pth")

    np.savetxt(os.path.join(os.getcwd(), 'result/runs', args.exp_name, "train_batch_loss.txt"), train_loss, fmt="%.4f")
    np.savetxt(os.path.join(os.getcwd(), 'result/runs', args.exp_name, "train_epoch_loss.txt"), train_epochs_loss, fmt="%.4f")
    np.savetxt(os.path.join(os.getcwd(), 'result/runs', args.exp_name, "Valid_epoch_accuracy.txt"), valid_epochs_acc, fmt="%.4f")
    with open(os.path.join(os.getcwd(), 'result/runs', args.exp_name, "evaluate_metrics.json"), 'w') as f:
        json.dump(evaluate_metrics, f, indent=2)


if __name__ == '__main__':
    random_seed(42):
    config = yaml.load(open("config.yml"), Loader=yaml.FullLoader)
    args = Edict(config)

    # ==== sava the parameter ====
    config_path = os.path.join(os.getcwd(), 'result/runs', args.exp_name)
    if not os.path.exists(config_path):
        os.makedirs(config_path, exist_ok=True)
    config_path = os.path.join(config_path, 'config.yml')
    with open(config_path, 'w', encoding="utf-8") as f:
        print("# run time: ", current_time(), file=f)
        yaml.dump(vars(args), f, allow_unicode=True)


    if args.data_use == "img":
        model = ImageModel(args.model_name, args.hidden_dims).to(args.device)
    elif args.data_use == "tab":
        model = TabularModel().to(args.device)
    elif args.data_use == "img_tab":
        model = Acute(args.model_name, args.hidden_dims).to(args.device)
    else:
        assert False, f"data_use error, not support {args.data_use}"

    if args.pre_tr:
        if hasattr(args, 'state_dict_path'):
            model.load_state_dict(torch.load(args.state_dict_path, map_location=args.device), strict=True)
    else:
        model.apply(init_weight)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)

    criterion = nn.CrossEntropyLoss()


    train_no, test_no = np.load(args.train_no_path), np.load(args.test_no_path)
    train_dataset = TotalDataset(args.cln_path, train_no)
    test_dataset = TotalDataset(args.cln_path, test_no, is_train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)

    train(model, train_dataloader, test_dataloader, criterion, optim, args.num_epochs, args)
