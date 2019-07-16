import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torch.utils.model_zoo as model_zoo

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torchvision import transforms, models


def tansf(x):
    im_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((0, 180)),
        transforms.ToTensor()
    ])
    x = im_aug(x)
    return x


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, _data, _label, _transform=None):
        self.imgs = _data
        self.labels = _label
        self.transform = _transform
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        return img, label


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.classes = num_classes
        self.fea = nn.Sequential(
            nn.Conv2d(1, 32, (5,5), 1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, (5,5), 1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, (3,3), 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3), 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2),
            nn.Dropout2d(0.25),
        )
        self.dense = nn.Sequential(
            nn.Linear(64*7*7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        out = self.fea(x)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        return out


def train(_train_data, _params):
    y_train = _train_data['label']
    x_train = _train_data.drop(labels=['label'], axis=1)
    x_train = pd.DataFrame(x_train, dtype=np.float) / 255
    x_train = x_train.values.reshape(-1, 28, 28, 1)
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    y_train = np.eye(N=10, dtype=np.float)[y_train]
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)

    device = torch.device('cuda')
    x_train = torch.from_numpy(x_train).to(device=device, dtype=torch.float)
    y_train = torch.from_numpy(y_train).to(device=device, dtype=torch.float)
    x_val = torch.from_numpy(x_val).to(device=device, dtype=torch.float)
    y_val = torch.from_numpy(y_val).to(device=device, dtype=torch.float)

    train_inputs = MyDataset(x_train, _label=y_train, _transform=tansf)
    val_inputs = MyDataset(x_val, y_val)
    train_loader = Data.DataLoader(train_inputs, batch_size=_params['batch_size'], shuffle=True)
    val_loader = Data.DataLoader(val_inputs, batch_size=_params['batch_size'])

    # model = Net().to(device)
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    num_fc = model.fc.in_features
    model.fc = nn.Linear(num_fc, 10)
    model.cuda()
    error = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=_params['lr_init'])
    lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=_params['lr_step_size'], gamma=_params['lr_step_rate'], last_epoch=-1)

    loss_dict = []
    step_dict = []
    acc_dict = []
    best_model = model
    best_acc = 0.0
    for epoch in range(_params['num_epochs']):
        for step, (img, label) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(img)
            loss = error(outputs, label)
            loss.backward()
            optimizer.step()
            lr_schedule.step()

            if step != 0 and step % _params['test_step'] == 0:
                correct = 0
                total = 0
                for val_img, val_label in val_loader:
                    val_outputs = model(val_img)
                    val_pred = torch.max(val_outputs.data, 1)[1]
                    true_label = torch.max(val_label, 1)[1]
                    total += len(true_label)
                    correct += (val_pred == true_label).sum()
                acc = correct.cpu().numpy() / float(total)
                loss_dict.append(loss.data)
                step_dict.append(_params['batch_size']*(epoch+1)+step+1)
                acc_dict.append(acc)
                print('Epoch: {}, Step: {}, Loss: {}, Acc: {}'.format(epoch, step, loss.item(), acc))
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
    torch.save(best_model, _params['model_name'])

    plt.plot(step_dict, loss_dict)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.show()

    # visualization accuracy 
    plt.plot(step_dict, acc_dict,color = "red")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.show()


def predict(_test_data, _params):
    _model = torch.load(_params['model_name']).cuda()
    _test_data = pd.DataFrame(_test_data, dtype=np.float) / 255
    _test_data = _test_data.values.reshape(-1, 28, 28, 1)
    _test_data = np.transpose(_test_data, (0, 3, 1, 2))
    test = torch.from_numpy(_test_data).to(device=torch.device('cuda'), dtype=torch.float)
    test_pred = np.empty((test.shape[0]))
    for test_index in range(0, test.shape[0], 100):
        test_outputs = _model(test[test_index: test_index+100])
        preds = torch.max(test_outputs.data, 1)[1]
        test_pred[test_index: test_index+100] = preds.cpu().numpy()
    pred_res = pd.DataFrame({ 'ImageId' : np.arange(1,test.shape[0]+1),
                            'Label' : test_pred.astype(int)})
    pred_res.set_index('ImageId', inplace=True)
    pred_res.to_csv(_params['res_name'])


if __name__ == "__main__":
    params = {
        'num_epochs': 10,
        'batch_size': 256,
        'lr_init': 0.001,
        'lr_step_size': 500,
        'lr_step_rate': 0.5,
        'test_step': 10,
        'model_name': 'my_solution2.pkl',
        'res_name': 'my_solution2.csv'
    }
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    train(train_data, params)
    predict(test_data, params)
