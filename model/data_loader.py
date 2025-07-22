import os
import numpy as np
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader


# Data loader and generator.
class Load_Dataset(Dataset):
    def __init__(self, dataset, configs, training_mode, batch_size, normalize=False, subset=False):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        x_train = dataset["samples"]
        y_train = dataset["labels"]

        if normalize:
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            x_train_reshape = x_train.reshape(x_train.shape[0], -1)
            x_train_reshape_norm = min_max_scaler.fit_transform(x_train_reshape).astype("float64")
            x_train_reshape_norm = x_train_reshape_norm.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
            x_train = torch.Tensor(x_train_reshape_norm).to(x_train.device)

        train_data = list(zip(x_train, y_train))
        np.random.shuffle(train_data)
        x_train, y_train = zip(*train_data)
        x_train, y_train = torch.stack(list(x_train), dim=0), torch.stack(list(y_train), dim=0)

        if len(x_train.shape) < 3:
            x_train = x_train.unsqueeze(1)
        x_train = x_train[:, :, :int(configs.fea_length)]

        if subset == True:
            subset_size = batch_size * 10
            x_train = x_train[:subset_size, :, :]
            y_train = y_train[:subset_size]

        if isinstance(x_train, np.ndarray):
            self.x_data = torch.from_numpy(x_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = x_train
            self.y_data = y_train

        self.len = x_train.shape[0]
        print("The " + self.training_mode + " dataset size is %s." % self.len)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(source_path, target_path, configs, normalize=False, subset=False):
    pretrain_dataset = torch.load(os.path.join(source_path, "train.pt"))
    finetune_dataset = torch.load(os.path.join(target_path, "train.pt"))
    test_dataset = torch.load(os.path.join(target_path, "test.pt"))

    # Load datasets.
    pretrain_dataset = Load_Dataset(pretrain_dataset, configs, training_mode="pre_train",
                                    batch_size=configs.source_batch_size, normalize=normalize, subset=True)
    finetune_dataset = Load_Dataset(finetune_dataset, configs, training_mode="fine_tune_test",
                                    batch_size=configs.target_batch_size, normalize=normalize, subset=subset)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode="fine_tune_test",
                                batch_size=configs.target_batch_size, normalize=normalize, subset=subset)

    # Dataloader.
    pretrain_loader = DataLoader(dataset=pretrain_dataset, batch_size=configs.source_batch_size,
                                 shuffle=True, drop_last=True, num_workers=0)
    finetune_loader = DataLoader(dataset=finetune_dataset, batch_size=configs.target_batch_size,
                                 shuffle=True, drop_last=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=configs.target_batch_size,
                             shuffle=True, drop_last=False, num_workers=0)

    return pretrain_loader, finetune_loader, test_loader
