import sys; sys.path.append("..")
import os
import os.path as osp
from pathlib import Path
import json

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict
from sklearn.metrics import roc_auc_score, confusion_matrix

from models.model import ImageModel, TabularModel, Acute
from data.Dataset import TotalDataset
from utils import metrics


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

        pred = F.softmax(logit, dim=1)
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


def evaluate_handler(args):
    model = None
    if args.data_use == "img":
        model = ImageModel(args.model_name, args.hidden_dims).to(args.device)
    elif args.data_use == "tab":
        model = TabularModel().to(args.device)
    elif args.data_use == "img_tab":
        model = Acute(args.model_name, args.hidden_dims).to(args.device)
    else:
        assert False, f"data_use error, not support {args.data_use}"

    test_no = np.load(args.test_no_path)
    test_dataset = TotalDataset(args.data_path, test_no, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    criterion = torch.nn.CrossEntropyLoss()

    evaluate_metrics = {}
    epoch_name = Path(osp.join(args.state_dict_dir, args.dir)).glob("*.pth")
    best_acc = np.NINF
    for name in epoch_name:
        "model_checkpoint_epoch11"
        args.state_dict = osp.join(args.state_dict_dir, args.dir, name)
        if name.stem == "model_checkpoint":
            epoch = -1
        else:
            epoch = int(name.stem[22:])
        if args.state_dict:
            model.load_state_dict(torch.load(args.state_dict, map_location=args.device), strict=True)
        else:
            raise Exception("No state_dict provided")
        loss, y_score, y_label = evaluate(test_dataloader, model, criterion, args)
        y_pred = torch.argmax(y_score, dim=1)
        y_truth = torch.argmax(y_label, dim=1)
        acc_total = torch.sum(y_pred == y_truth).item() / len(y_truth)

        # region Calculate metrics
        # hetero Lmyp disease
        y_score_malignant, y_malignant = y_score[:, 2].cpu().numpy(), y_label[:, 2].cpu().numpy()
        auc_hetero = roc_auc_score(y_malignant, y_score_malignant)
        acc_hetero, prec_hetero, rec_hetero, f1_score_hetero = metrics(y_pred.cpu().numpy() == 2, y_malignant)

        # acute Lmyp disease
        y_score_malignant, y_malignant = y_score[:, 0].cpu().numpy(), y_label[:, 0].cpu().numpy()
        auc_acute = roc_auc_score(y_malignant, y_score_malignant)
        acc_acute, prec_acute, rec_acute, f1_score_acute = metrics(y_pred.cpu().numpy()==0, y_malignant)
        
        # normal Lmyp disease
        y_score_malignant, y_malignant = y_score[:, 1].cpu().numpy(), y_label[:, 1].cpu().numpy()
        auc_normal = roc_auc_score(y_malignant, y_score_malignant)
        acc_normal, prec_normal, rec_normal, f1_score_normal = metrics(y_pred.cpu().numpy()==1, y_malignant)
        
        #endregion
        save_path = os.path.join(args.save_dir, args.dir, f"epoch{epoch}.txt")
        np.savetxt(save_path,
                    np.concatenate((y_score.transpose(0, 1).cpu(), y_label.transpose(0, 1).cpu()), axis=0),
                    delimiter=',',
                    fmt="%.4f")
        print(args.dir, f"epoch{epoch}")
        print(confusion_matrix(y_truth, y_pred))
        print('-' * 20)


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

            save_path = os.path.join(args.save_dir, args.dir, f"best_epoch.txt")
            np.savetxt(save_path,
                       np.concatenate((y_score.transpose(0, 1).cpu(), y_label.transpose(0, 1).cpu()), axis=0),
                       delimiter=',',
                       fmt="%.4f")

    with open(os.path.join(args.save_dir, args.dir, "evaluate_metrics.json"), 'w') as f:
        json.dump(evaluate_metrics, f, indent=2)


def evaluate_handler_best(args):
    model = None
    if args.data_use == "img":
        model = ImageModel(args.model_name, args.hidden_dims).to(args.device)
    elif args.data_use == "tab":
        model = TabularModel().to(args.device)
    elif args.data_use == "img_tab":
        model = Acute(args.model_name, args.hidden_dims).to(args.device)
    else:
        assert False, f"data_use error, not support {args.data_use}"

    test_no = np.load(args.test_no_path)
    test_dataset = TotalDataset(args.data_path, test_no, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    criterion = torch.nn.CrossEntropyLoss()

    evaluate_metrics = {}
    epoch_name = Path(osp.join(args.state_dict_dir, args.dir, "model_checkpoint.pth"))

    args.state_dict = osp.join(args.state_dict_dir, args.dir, epoch_name)
    if args.state_dict:
        model.load_state_dict(torch.load(args.state_dict, map_location=args.device), strict=True)
    else:
        raise Exception("No state_dict provided")
    loss, y_score, y_label = evaluate(test_dataloader, model, criterion, args)
    y_pred = torch.argmax(y_score, dim=1)
    y_truth = torch.argmax(y_label, dim=1)
    acc_total = torch.sum(y_pred == y_truth).item() / len(y_truth)

    # region Calculate metrics
    # hetero Lmyp disease
    y_score_malignant, y_malignant = y_score[:, 2].cpu().numpy(), y_label[:, 2].cpu().numpy()
    auc_hetero = roc_auc_score(y_malignant, y_score_malignant)
    acc_hetero, prec_hetero, rec_hetero, f1_score_hetero = metrics(y_pred.cpu().numpy() == 2, y_malignant)

    # acute Lmyp disease
    y_score_malignant, y_malignant = y_score[:, 0].cpu().numpy(), y_label[:, 0].cpu().numpy()
    auc_acute = roc_auc_score(y_malignant, y_score_malignant)
    acc_acute, prec_acute, rec_acute, f1_score_acute = metrics(y_pred.cpu().numpy()==0, y_malignant)
    
    # normal Lmyp disease
    y_score_malignant, y_malignant = y_score[:, 1].cpu().numpy(), y_label[:, 1].cpu().numpy()
    auc_normal = roc_auc_score(y_malignant, y_score_malignant)
    acc_normal, prec_normal, rec_normal, f1_score_normal = metrics(y_pred.cpu().numpy()==1, y_malignant)
    
    #endregion
    save_path = os.path.join(args.save_dir, args.dir, f"best_epoch.txt")
    np.savetxt(save_path,
                np.concatenate((y_score.transpose(0, 1).cpu(), y_label.transpose(0, 1).cpu()), axis=0),
                delimiter=',',
                fmt="%.4f")
    print(args.dir, f"best epoch")
    print(confusion_matrix(y_truth, y_pred))
    print('-' * 20)

    evaluate_metrics["epoch"] = -1
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

    with open(os.path.join(args.save_dir, args.dir, "evaluate_metrics.json"), 'w') as f:
        json.dump(evaluate_metrics, f, indent=2)


if __name__ == "__main__":

    config = {
        "data_use": "img_tab",
        "model_name": "resnet18",
        "dir": None,

        "hidden_dims": [3],
        "test_no_path": None
        "data_path": None,
        "device": "cuda:3",
        "state_dict_dir": None,
        "save_dir": None
    }

    args = EasyDict(config)
    
    args.dir = f"10_12_forward_gate_resnet18_xcg_200e_repeat_fold5"
    if not osp.exists(osp.join(args.save_dir, args.dir)):
        os.makedirs(osp.join(args.save_dir, args.dir))
    evaluate_handler_best(args)

