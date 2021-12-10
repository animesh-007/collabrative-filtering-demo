import torch
import torch.utils.data as data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import pandas as pd


class Events_Dataset(data.Dataset):
    def __init__(self, args, mode="Train"):

        self.mode = mode
        # self.training = copy.deepcopy(hp.training)

        data = pd.read_csv(f"{args.path}.csv")

        # split train and validation before encoding
        np.random.seed(3)
        msk = np.random.rand(len(data)) < 0.8
        train = data[msk].copy()
        val = data[~msk].copy()

        # train["ratings"] = np.ones(len(train))
        # val["ratings"] = np.ones(len(val))

        # encoding the train and validation data
        df_train = encode_data(train)
        df_val = encode_data(val, train)

        num_users = len(df_train.actor.unique())
        num_items = len(df_train.object_id.unique())
        
        print(num_users, num_items)

        
        # print(get_all_classes)
        # if self.training == "photo":
        self.train_num_users, self.train_num_items = torch.LongTensor(df_train.actor.values), torch.LongTensor(df_train.object_id.values)

        self.test_num_users, self.test_num_items = torch.LongTensor(df_val.actor.values), torch.LongTensor(df_val.object_id.values)

        print("Total Training users {}".format(len(self.train_num_users)))
        print("Total Testing users {}".format(len(self.test_num_users)))
        print("Total Training items {}".format(len(self.train_num_items)))
        print("Total Testing items {}".format(len(self.test_num_items)))

    def __getitem__(self, item):

        if self.mode == "Train":

            user = self.train_num_users[item]
            user_item = self.train_num_items[item]

            sample = {
                "user": user,
                "user_item": user_item,
                "ratings": torch.FloatTensor([1]),
            }
            return sample

        else:

            user = self.test_num_users[item]
            user_item = self.test_num_items[item]

            sample = {
                "user": user,
                "user_item": user_item,
                "ratings": torch.FloatTensor([1]),
            }
            return sample

    def __len__(self):
        if self.mode == "Train":
            return len(self.train_num_users)
        elif self.mode == "Test":
            return len(self.test_num_users)

# here is a handy function modified from fast.ai
def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids. 
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)

def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["actor", "object_id"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df

def get_dataloader(args):

    dataset_Train = Events_Dataset(args, mode="Train")
    dataset_Test = Events_Dataset(args, mode="Test")

    dataloader_Train = data.DataLoader(
        dataset_Train,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=8,
    )

    dataloader_Test = data.DataLoader(
        dataset_Test,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=8,
    )

    return dataloader_Train, dataloader_Test, len(dataloader_Train.dataset.train_num_users), len(dataloader_Train.dataset.train_num_items)
