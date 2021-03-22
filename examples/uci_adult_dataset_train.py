# UCI dataset
# https://archive.ics.uci.edu/ml/datasets/adult

import copy 
import random
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn import metrics as skm
from sklearn import  svm, tree, linear_model
from fairlearn import metrics as flm
from fairtorch import ConstraintLoss, DemographicParityLoss, EqualiedOddsLoss, AdversaryNet, AdversarialDebiasingLoss
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from torch import optim
import fairtorch
import pprint

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(2020)
torch.random.manual_seed(2020)

class SensitiveDataset(Dataset):
    def __init__(self, x, y, sensitive):
        if isinstance(x, pd.DataFrame):
            x = torch.from_numpy(x.values)
        if isinstance(y, pd.DataFrame):
            y = torch.from_numpy(y.values)
        else:
            y = torch.Tensor(y)
        if isinstance(x, torch.Tensor):
            self.x = x.float()
        if isinstance(y, torch.Tensor):
            self.y = y.float()
        # self.y = np.ones(shape=y.shape).astype(np.float32)
        sensitive_categories = sensitive.unique()
        # print(sencat)
        self.category_to_index_dict = dict(
            zip(list(sensitive_categories), range(len(sensitive_categories)))
        )
        self.index_to_category_dict = dict(
            zip(range(len(sensitive_categories)), list(sensitive_categories))
        )
        self.sensitive = torch.Tensor(sensitive).long()
        self.sensitive_ids = [
            self.category_to_index_dict[i] for i in self.sensitive.numpy().tolist()
        ]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx].reshape(-1), self.sensitive_ids[idx]


def train_model(model, device, criterion, constraints, data_loader, optimizer, max_epoch=1):
    for epoch in range(max_epoch):
        for i, data in enumerate(data_loader):
            x, y, sensitive_features = data
            x = x.to(device)
            y = y.to(device)
            sensitive_features = sensitive_features.to(device)
            optimizer.zero_grad()
            # print(x.device, y.device, sensitive_features.device)
            # print(x.shape, y.shape, sensitive_features.shape)

            logit = model(x)
            assert isinstance(logit, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            # print(x.device, y.device, sensitive_features.device, logit.device)

            loss = criterion(logit, y)
            if constraints:
                penalty = constraints(x, logit, sensitive_features, y)
                # print(penalty.requires_grad)
                loss = loss + penalty
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, device, criterion, data_loader):
    model.eval()
    y_true = []
    y_pred = []
    y_out = []
    sensitives = []
    for i, data in enumerate(data_loader):
        x, y, sensitive_features = data
        # print(x, y, sensitive_features)
        # sys.exit()
        x = x.to(device)
        y = y.to(device)
        sensitive_features = sensitive_features.to(device)
        with torch.no_grad():
            logit = model(x)
        # logit : binary prediction size=(b, 1)
        bina = (torch.sigmoid(logit) > 0.5 ).float()
        y_true += y.cpu().tolist()
        y_pred += bina.cpu().tolist()
        y_out += torch.sigmoid(logit).tolist()
        sensitives += sensitive_features.cpu().tolist()
    # print(sensitives)
    result = {}
    result["acc"] = skm.accuracy_score(y_true, y_pred)
    result["f1score"] = skm.f1_score( y_true, y_pred)
    try:
        result["AUC"] = skm.roc_auc_score(y_true, y_out)
    except :
        result["AUC"] = None
    # print("samples, ", len(y_true), len(y_pred), len(sensitives))
    result['DP'] = {
        "diff":flm.demographic_parity_difference(
            y_true,
            y_pred, 
            sensitive_features= sensitives),
        "ratio": flm.demographic_parity_ratio(
            y_true,
            y_pred, 
            sensitive_features= sensitives),
    }
    result["EO"] = {
        "diff":flm.equalized_odds_difference(
            y_true,
            y_pred, 
            sensitive_features= sensitives),
        "ratio": flm.equalized_odds_ratio(
            y_true,
            y_pred, 
            sensitive_features= sensitives),
    }
    flm_frame = flm.MetricFrame({"selection_rate":flm.selection_rate, "true_positive_rate": flm.true_positive_rate},
            y_true,
            y_pred, 
            sensitive_features= sensitives)
    result["frame"] = flm_frame.by_group
    return result


def train_and_eval(constraints=None):
    print(" ")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    #print(device)
    

    data = fetch_openml(data_id=1590, as_frame=True)

    X = pd.get_dummies(data.data)
    y_true = (data.target == '>50K') * 1

    # chance rate
    print("chance rate= ", y_true.sum()/y_true.shape[0])
    sex = data.data['sex']
    sex_bin = sex == "Male"
    # print(sex_bin)
    feature_dim = X.shape[1]
    dim_condition=2 
    dataset = SensitiveDataset(X, y_true, sex_bin)

    train_size = int(len(dataset) * 0.6)
    valid_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - valid_size
    print(f"n_train={train_size}, n_valid={valid_size}, n_test={test_size}")
    train_dataset,valid_dataset,  test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )
    #print(device)
    model = nn.Sequential(nn.Linear(feature_dim, 32), nn.Dropout(), nn.LeakyReLU(), nn.Linear(32, 1), nn.Dropout())
    model.to(device) 
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001,  )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    max_epoch = 1000
    max_auc = 0
    aucs = []
    for i in range(max_epoch):
        model = train_model(
            data_loader=train_loader,
            device=device,
            model=model,
            criterion=criterion,
            constraints=constraints,
            optimizer=optimizer,
            max_epoch=1)
        result = evaluate_model(model, device, criterion, valid_loader)
        aucs.append(result['AUC'])
        if i > 3:
            if np.mean(aucs[-3:]) < max_auc :
                print(f"train end at epoch={i+1}")
                break
        if result['AUC'] > max_auc:
            max_auc = result['AUC']
    result = evaluate_model(model, device, criterion, test_loader)
    print("constraint: ", constraints)
    if constraints and isinstance(constraints, AdversarialDebiasingLoss):
        print(f"parity: {constraints.parity}, n_iter: {constraints.n_iter}")
    pprint.pprint(result)

    if constraints :
        print("alpha: ", constraints.alpha)
    # make parity condition
    if isinstance(constraints, DemographicParityLoss) :
        parity = "DP"
    elif isinstance(constraints, EqualiedOddsLoss):
        parity = "EO"
    elif isinstance(constraints, AdversarialDebiasingLoss):
        parity = constraints.parity
    else:
        parity= None
    return result

def main():
    train_and_eval(None)
    train_and_eval(DemographicParityLoss(penalty="penalty", alpha=1))
    train_and_eval(DemographicParityLoss(penalty="penalty", alpha=10))
    train_and_eval(DemographicParityLoss(penalty="penalty", alpha=100))
    train_and_eval(DemographicParityLoss(penalty="exact_penalty", alpha=1))
    train_and_eval(DemographicParityLoss(penalty="exact_penalty", alpha=10))
    train_and_eval(DemographicParityLoss(penalty="exact_penalty", alpha=100))
    train_and_eval(EqualiedOddsLoss(penalty="penalty", alpha=1))
    train_and_eval(EqualiedOddsLoss(penalty="penalty", alpha=10))
    train_and_eval(EqualiedOddsLoss(penalty="exact_penalty", alpha=10))     
    train_and_eval( constraints=AdversarialDebiasingLoss(parity="DP", n_layers=1, alpha=1) )
    train_and_eval( constraints=AdversarialDebiasingLoss(parity="DP", n_layers=1, alpha=10) )
    train_and_eval( constraints=AdversarialDebiasingLoss(parity="DP", n_layers=1, alpha=100))
    train_and_eval( constraints=AdversarialDebiasingLoss(parity="EO", n_layers=1, alpha=10) )
    train_and_eval( constraints=AdversarialDebiasingLoss(parity="DP", n_layers=4, alpha=1) )
    train_and_eval( constraints=AdversarialDebiasingLoss(parity="DP", n_layers=4, alpha=10) )
    train_and_eval( constraints=AdversarialDebiasingLoss(parity="DP", n_layers=4, alpha=100) )

if __name__=='__main__':
    main()