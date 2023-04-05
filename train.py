import numpy as np
import pandas as pd

import torch
import pickle
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import confusion_matrix

from net import FusionNet


class PoseDataset(Dataset):
    def __init__(self, behavior_name):
        self.behavior_name = behavior_name
        self.dataset = pd.read_json(
            "./dataset/result/final/" + self.behavior_name + ".json"
        )

    def __getitem__(self, idx):
        final_dataset = np.append(
            torch.tensor(self.dataset["body_point"][idx], dtype=torch.float32),
            torch.tensor(self.dataset["face_image"][idx], dtype=torch.float32),
        )
        labels = torch.tensor(self.dataset["behavior_index"][idx], dtype=torch.float32)
        return final_dataset, labels

    def __len__(self):
        return self.dataset.shape[0]


def dataset_split(pose_dataset):
    train_dataset, val_dataset, test_dataset = random_split(
        pose_dataset, [0.7, 0.2, 0.1], torch.Generator().manual_seed(42)
    )
    return train_dataset, val_dataset, test_dataset


drink_dataset = PoseDataset("drink")
listen_dataset = PoseDataset("listen")
phone_dataset = PoseDataset("phone")
trance_dataset = PoseDataset("trance")
write_dataset = PoseDataset("write")

drink_train_dataset, drink_val_dataset, drink_test_dataset = dataset_split(
    drink_dataset
)
listen_train_dataset, listen_val_dataset, listen_test_dataset = dataset_split(
    listen_dataset
)
phone_train_dataset, phone_val_dataset, phone_test_dataset = dataset_split(
    phone_dataset
)
trance_train_dataset, trance_val_dataset, trance_test_dataset = dataset_split(
    trance_dataset
)
write_train_dataset, write_val_dataset, write_test_dataset = dataset_split(
    write_dataset
)

train_dataset = (
    drink_train_dataset
    + listen_train_dataset
    + phone_train_dataset
    + trance_train_dataset
    + write_train_dataset
)
val_dataset = (
    drink_val_dataset
    + listen_val_dataset
    + phone_val_dataset
    + trance_val_dataset
    + write_val_dataset
)
test_dataset = (
    drink_test_dataset
    + listen_test_dataset
    + phone_test_dataset
    + trance_test_dataset
    + write_test_dataset
)

net = FusionNet()


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)
batch_size, lr, num_epochs = 64, 0.001, 200
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    [
        {"params": net.pose_net.parameters(), "lr": 0.1},
        {"params": net.face_net.parameters(), "lr": 0.001},
        {"params": net.head_net.parameters(), "lr": 0.001},
        {"params": net.fusion_net.parameters(), "lr": 0.001},
    ],
    lr=lr,
)

y_loss = {}
y_loss["train"] = []
y_loss["val"] = []
y_acc = {}
y_acc["train"] = []
y_acc["val"] = []
y_test = {}
y_test["cm"] = []


def train(epoch):
    sum_loss = 0.0
    train_size = 0
    correct = 0
    total = 0
    net.train(True)
    for data in train_loader:
        inputs, labels = data
        now_batch_size, _ = inputs.shape
        train_size += now_batch_size
        labels = labels.long()
        # inputs, labels = inputs.to('cuda'), labels.long().cuda()
        optimizer.zero_grad()

        # FusionNet
        pose = inputs[:, 0:272].reshape(-1, 1, 16, 17)
        face = inputs[:, 275:].reshape(-1, 3, 30, 30) / 255
        head = inputs[:, 272:275].reshape(-1, 1, 3)
        outputs = net(pose, face, head)

        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = cost(outputs, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item() * now_batch_size

    print(train_size)
    y_loss["train"].append(sum_loss / train_size)
    y_acc["train"].append(correct / total)
    print(
        "epoch:%d\ntrain: loss:%.3f acc:%.3f"
        % (epoch + 1, sum_loss / train_size, 100 * correct / total)
    )


def val():
    sum_loss = 0.0
    val_size = 0
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    net.train(False)
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            now_batch_size, _ = inputs.shape
            val_size += now_batch_size
            labels = labels.long()
            y_true.append(labels)
            # inputs, labels = inputs.to('cuda'), labels.to('cuda')

            # FusionNet
            pose = inputs[:, 0:272].reshape(-1, 1, 16, 17)
            face = inputs[:, 275:].reshape(-1, 3, 30, 30) / 255
            head = inputs[:, 272:275].reshape(-1, 1, 3)
            outputs = net(pose, face, head)

            _, predicted = torch.max(outputs.data, dim=1)
            y_pred.append(predicted.to("cpu"))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = cost(outputs, labels)
            sum_loss += loss.item() * now_batch_size

    print(val_size)
    y_loss["val"].append(sum_loss / val_size)
    y_acc["val"].append(correct / total)

    print("val: loss:%.3f acc:%.3f" % (sum_loss / val_size, 100 * correct / total))
    return y_true, y_pred


def test():
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    net.train(False)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            y_true.append(labels)
            # inputs, labels = inputs.to('cuda'), labels.to('cuda')

            # FusionNet
            pose = inputs[:, 0:272].reshape(-1, 1, 16, 17)
            face = inputs[:, 275:].reshape(-1, 3, 30, 30) / 255
            head = inputs[:, 272:275].reshape(-1, 1, 3)
            outputs = net(pose, face, head)

            _, predicted = torch.max(outputs.data, dim=1)
            y_pred.append(predicted.to("cpu"))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return y_true, y_pred


for epoch in range(num_epochs):
    train(epoch)
    y_true, y_pred = val()
    print(
        confusion_matrix(
            torch.cat(y_true),
            torch.cat(y_pred),
        )
    )
    print("=========================")
    y_true, y_pred = test()

    y_test["cm"].append(
        confusion_matrix(
            torch.cat(y_true),
            torch.cat(y_pred),
        )
    )

np.save("./result/y_loss.npy", pickle.dumps(y_loss))
np.save("./result/y_acc.npy", pickle.dumps(y_acc))
np.save("./result/y_test.npy", pickle.dumps(y_test))
