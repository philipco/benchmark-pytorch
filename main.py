"""
Created by Philippenko, 16th February 2022.
"""

from Dataset import Dataset
from Models import LeNet

from Trainer import Training

BATCH_SIZE = 128

if __name__ == '__main__':

    ### Getting the dataset
    dataset = Dataset(batch_size = BATCH_SIZE, dataset_name="cifar10")
    train_loader, testloader, time_data_loading = dataset.get_loaders()

    ### Intialization of the trainer
    trainer = Training(LeNet, train_loader, testloader)

    with open(trainer.logs_file, 'a') as f:
        print("Time loading datasets: {:.2e}s".format(time_data_loading), file=f)

    ### Running the training
    run_logger = trainer.run_training()

    ### Plotting the train loss and the accuracy
    run_logger.plot()

    print('Finished Training')


