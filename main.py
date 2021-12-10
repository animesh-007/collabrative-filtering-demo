import torch
from model import Collab_Model
from dataset import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collaborative filtering")  

    parser.add_argument(
        "--base_dir", type=str, default=os.getcwd(), help="In order to access from condor"
    )
    parser.add_argument(
        "--saved_models", type=str, default="./models", help="Saved models directory"
    )
    parser.add_argument(
        "--path", type=str, default="events", help="Events dataset"
    )
    parser.add_argument("--batchsize", type=int, default=512)
    parser.add_argument("--nThreads", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--eval_freq_iter", type=int, default=2)
    parser.add_argument("--print_freq_iter", type=int, default=1)

    args = parser.parse_args()
    dataloader_Train, dataloader_Test, num_users, num_items = get_dataloader(args)

    model = Collab_Model(args,num_users, num_items)
    model.to(device)
    # best_accuracy = 0

    # os.makedirs(hp.saved_models, exist_ok=True)

    for epoch in range(args.max_epoch):

        for i, batch in enumerate(dataloader_Train):
            # save_image(batch["image"], "triplet_pair.png")
            loss = model.train_model(batch)

            if i % args.print_freq_iter == 0:
                print(
                    "Epoch: {}, Iter: {}, Loss: {}".format(
                        epoch, i, loss
                    )
                )

            if epoch % args.eval_freq_iter == 0:
                with torch.no_grad():
                    model.evaluate(dataloader_Test)
                # if accuracy > best_accuracy:
                #     best_accuracy = accuracy
                #     torch.save(
                #         model.state_dict(),
                #         os.path.join(
                #             hp.saved_models, "model_best_" + str(hp.training) + ".pth"
                #         ),
                #     )
