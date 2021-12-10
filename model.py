from network import CollabFNet
from torch import optim
import torch
import time
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Collab_Model(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(Collab_Model, self).__init__()
        self.Network = CollabFNet(num_users, num_items, emb_size=100)
        self.train_params = self.parameters()
        self.optimizer = optim.Adam(self.train_params, args.learning_rate)
        self.loss = nn.MSELoss()
        self.args = args

    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        output = self.Network(batch["user"].to(device), batch["user_item"].to(device))
        loss = self.loss(output, batch["ratings"].to(device))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, dataloader_Test):
        self.eval()
        correct = 0
        test_loss = 0
        start_time = time.time()
        for i_batch, batch in enumerate(dataloader_Test):

            output = self.Network(batch["user"].to(device), batch["user_item"].to(device))
            test_loss += self.loss(output, batch["ratings"].to(device)).item()
            prediction = output.argmax(dim=1, keepdim=True).to("cpu")
            # correct += prediction.eq(batch["ratings"].view_as(prediction)).sum().item()

        test_loss /= len(dataloader_Test.dataset)
        # accuracy = 100.0 * correct / len(dataloader_Test.dataset)

        # print(
        #     "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Time_Takes: {}\n".format(
        #         test_loss,
        #         correct,
        #         len(dataloader_Test.dataset),
        #         accuracy,
        #         (time.time() - start_time),
        #     )
        # )

        print(
            "\nTest set: Average loss: {:.4f}, Time_Takes: {}\n".format(
                test_loss,
                (time.time() - start_time),
            )
        )

        # return accuracy
