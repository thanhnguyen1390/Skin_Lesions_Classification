import pandas as pd
import os
import sys
import numpy as np
import time
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from PIL import Image
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')  # use agg backend for server
import matplotlib.pyplot as plt
import smtplib, ssl
from sklearn.metrics import confusion_matrix


##################
#     DATASET    #
##################
class Mnist_Ham10000(Dataset):
    '''MNIST_HAM10000 Dataset'''

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.image_dir = '../data/images'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_dir, self.df['image_id'][index] + '.jpg'))
        # image = image.resize((input_size, input_size), Image.ANTIALIAS)
        label = torch.tensor(self.df['cell_type_idx'][index])

        if self.transform:
            image = self.transform(image)

        return image, label

##################
#     NETWORK    #
##################
class trainer():
    def __init__(self, model, optimizer, loss, scheduler, classes, train_set, test_set, model_dir):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.train_set = train_set
        self.test_set = test_set
        self.classes = classes
        self.model_dir = model_dir
        self.train_time = 0

    def load_data(self, batch_size):
        # Function to load data into DataLoader.
        self.batch_size = batch_size
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False)

    def train(self, epochs):
        # Function to train and validate the network
        num_classes = len(self.classes)
        use_cuda = torch.cuda.is_available()

        if use_cuda:
            self.model.cuda()

        # track the training loss at each iteration and training & testing accuracy at each epoch
        self.running_loss = np.zeros((epochs, len(self.train_loader)))
        self.running_train_acc = np.zeros(epochs)
        self.running_test_acc = np.zeros(epochs)

        # Track the last epoch prediction results
        label_predict = []
        label_actual = []

        # Train
        start_time = time.time()
        for epoch in range(epochs):
            train_correct = 0
            test_correct = 0
            self.scheduler.step()
            self.model.train()
            for i, data in enumerate(self.train_loader):
                images, labels = data
                if use_cuda:
                    images = Variable(images).cuda()
                    labels = Variable(labels).cuda()

                self.optimizer.zero_grad()
                # Forward
                output = self.model(images)
                # Calculate loss
                err = self.loss(output, labels)
                # Backward
                err.backward()
                # Update weights
                self.optimizer.step()

                _, train_predict = torch.max(output.data, 1)

                train_correct += (train_predict.cpu().numpy() == labels.cpu().numpy()).sum()

                # show training loss at every 100 data points
                if (i + 1) % 100 == 0:
                    print('Training loss at epoch {} of {}, step {} of {}: {:.4f}'.format(
                        epoch + 1, epochs, (i + 1), len(self.train_loader), err.item()
                    ))

                # add to running loss
                self.running_loss[epoch, i] = err.item()

            # accuracy at each epoch: add to tracking
            self.running_train_acc[epoch] = train_correct / len(self.train_set) * 100

            self.model.eval()

            for data in self.test_loader:
                images, labels = data
                if use_cuda:
                    images = Variable(images).cuda()

                output = self.model(images)
                _, test_predict = torch.max(output.data, 1)

                test_correct += (test_predict.cpu().numpy() == labels.cpu().numpy()).sum()

                if epoch == epochs - 1:
                    for i in range(len(labels)):
                        label_predict.append(test_predict.cpu().numpy()[i])
                        label_actual.append(labels.cpu().numpy()[i])

            # accuracy at each epoch: add to tracking, print
            self.running_test_acc[epoch] = test_correct / len(self.test_set) * 100

            print('Accuracy of network on test set at epoch {} of {}: {}/{} = {:.2f}%'.format(
                epoch + 1, epochs, test_correct, len(self.test_set), test_correct / len(self.test_set) * 100
            ))

            if epoch == epochs - 1:
                pd.DataFrame(list(zip(label_actual, label_predict)), columns=['Actual', 'Predicted']).to_csv(os.path.join(self.model_dir, 'test_result.csv'), index=False)
                self.confusion_matrix = confusion_matrix(label_actual, label_predict, labels=range(num_classes))

        end_time = time.time()
        self.train_time = end_time - start_time

    def save_model(self):
        # Function to save model for later use (continue training or predicting new data)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'model.pt'))


    def generate_graphs(self):
        # Function to generate necessary model graphs
        num_classes = len(self.classes)
        self.running_loss = self.running_loss.reshape(-1)

        # plot running loss
        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.plot(self.running_loss, alpha=0.7)
        # label x axis with epochs using formatter
        ax.set_xticklabels(['{:0.0f}'.format(i) for i in ax.get_xticks() / (len(self.train_set) / self.batch_size)])
        ax.set_title(self.model._get_name() + ' - Training Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        fig.savefig(os.path.join(self.model_dir, 'training_loss.png'))
        print('Training loss plot saved.')

        # plot comparison of testing and training accuracy
        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.plot(self.running_test_acc, label='Testing Accuracy', alpha=0.8)
        ax.plot(self.running_train_acc, label='Training Accuracy', alpha=0.8)
        ax.set_ylim(0, 100)
        ax.set_xticklabels('{:0.0f}'.format(i + 1) for i in ax.get_xticks())
        ax.set_title(self.model._get_name() + ' - Predication Accuracy')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Epoch')
        ax.legend(loc='lower right', fontsize='medium')
        fig.savefig(os.path.join(self.model_dir, 'testing_accuracy.png'))
        print('Testing Accuracy plot saved.')

        # confusion matrix of predictions

        # calculate accuracy and misclassification rate
        accuracy = self.confusion_matrix.diagonal().sum() / self.confusion_matrix.sum() * 100
        wrong = 100 - accuracy
        print('Accuracy rate: {:.2f}%\nMisclassifcation rate: {:.2f}%'.format(accuracy, wrong))

        # plot confusion matrix
        print(self.confusion_matrix)
        confusion_matrix_norm = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[num_classes, num_classes])
        ax.matshow(confusion_matrix_norm, cmap='summer', vmin=confusion_matrix_norm.min(),
                   vmax=confusion_matrix_norm.max())
        ax.set_xlim(-1, 7)
        ax.set_ylim(7, -1)  # reverse order
        ax.xaxis.tick_bottom()
        ax.set_xticklabels(self.classes, rotation=90, size='x-small')
        ax.set_xticks([i for i in range(num_classes)])
        ax.set_yticklabels(self.classes, size='x-small')
        ax.set_yticks([i for i in range(num_classes)])
        ax.set_xlabel('Actual', size='small')
        ax.set_ylabel('Predicted', size='small')
        ax.set_title(self.model._get_name() + ' - Confusion Matrix')

        # label cells
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(i, j, '{:.2f}%'.format(confusion_matrix_norm[j, i] * 100), ha='center', va='center', color='black')

        fig.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'), bbox_inches='tight')
        print('Confusion matrix plot saved.')

##################
#     EMAIL      #
##################
def send_email(message):
    # Function to send notification email
    port = 587
    smtp_sever = 'smtp.gmail.com'
    sender_email = '' #Sender email
    receiver_email = '' #Receiver email
    password = '' #Password

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_sever, port) as server:
        server.starttls(context=context)
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

# Env setup
plt.style.use(['classic'])
np.random.seed(13)
torch.manual_seed(13)

# Load metadata
data_dir = '../data'
metadata_filename = "HAM10000_metadata_cleaned.csv"

metadata = pd.read_csv(os.path.join(data_dir, metadata_filename))

# metadata = metadata.sample(1000)

classes = metadata[['cell_type', 'cell_type_idx']].sort_values('cell_type_idx').drop_duplicates()['cell_type'].as_matrix()

# Split into Train and Test sets
train_data, test_data = train_test_split(metadata, test_size=0.2, random_state=13, stratify=metadata.cell_type_idx.values)
train_data = train_data.reset_index()
test_data = test_data.reset_index()

# Plot Train and Test cell type distribution
train_data.cell_type.value_counts(sort=True, ascending=False).plot(kind="barh", title="Train Distribution", fontsize=8,
                                                                   rot= 0, figsize=(10,5))
plt.savefig('../plots/train_data_cell_type.png', bbox_inches='tight')
plt.clf()

test_data.cell_type.value_counts(sort=True, ascending=False).plot(kind="barh", title="Test Distribution", fontsize=8,
                                                                  rot= 0, figsize=(10,5))
plt.savefig('../plots/test_data_cell_type.png', bbox_inches='tight')
plt.clf()

# Export train and test set for later data visualization
train_data.to_csv('../models/train_data.csv', index=False)
test_data.to_csv('../models/test_data.csv', index=False)

# Hyper Parameters
learning_rate = 0.0001
decay_rate = 0.1
epochs = 20
input_size = 224
batch_size = 10

# Add class weight to make the models more sensitive to melanoma
class_weight = [3 if lesion_class == 'Melanoma' else 1 for lesion_class in classes]
class_weight_ts = torch.FloatTensor(class_weight)
if torch.cuda.is_available():
    class_weight_ts = class_weight_ts.cuda()

# Transformation and Data Augmentation
train_transform = transforms.Compose([
    transforms.CenterCrop(input_size + 10),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = Mnist_Ham10000(train_data, transform=train_transform)
test_set = Mnist_Ham10000(test_data, transform=test_transform)

# ##############################################################
# #                    RESNET18                                #
# ##############################################################
model_dir = '../models/resnet18'

model = models.resnet18(pretrained=True)

# Transfer learning: Freeze all convolutional layers and only train the classifier layer which is the last linear layer
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=decay_rate)

sys.stdout = open(os.path.join(model_dir, 'output.txt'), 'w')

print(model)

resnet18_trainer = trainer(model, optimizer, loss, scheduler, classes, train_set, test_set, model_dir)
resnet18_trainer.load_data(batch_size)
print("Start Training Resnet18")
resnet18_trainer.train(epochs)
print('Training time: {:0.1f} seconds.'.format(resnet18_trainer.train_time))
resnet18_trainer.save_model()
resnet18_trainer.generate_graphs()

print('End Training Resnet18')

message = """\
Subject: Training completed

Training ResNet18 model completed.
"""

# send_email(message)


# ##############################################################
# #                    RESNET18 WITH WEIGHT                    #
# ##############################################################
model_dir = '../models/resnet18_weight'

model = models.resnet18(pretrained=True)

# Transfer learning: Freeze all convolutional layers and only train the classifier layer which is the last linear layer
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = torch.nn.CrossEntropyLoss(weight=class_weight_ts)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=decay_rate)

sys.stdout = open(os.path.join(model_dir, 'output.txt'), 'w')

print(model)

resnet18_trainer = trainer(model, optimizer, loss, scheduler, classes, train_set, test_set, model_dir)
resnet18_trainer.load_data(batch_size)
print("Start Training Resnet18 with weight ")
resnet18_trainer.train(epochs)
print('Training time: {:0.1f} seconds.'.format(resnet18_trainer.train_time))
resnet18_trainer.save_model()
resnet18_trainer.generate_graphs()

print('End Training Resnet18 with weight ')

message = """\
Subject: Training completed

Training ResNet18 with weight model completed.
"""

# send_email(message)

# ##############################################################
# #                    ALEXNET                                 #
# ##############################################################
model_dir = '../models/alexnet'

model = models.alexnet(pretrained=True)

# Transfer learning: Freeze all convolutional layers and only train the last liner layer of classifier layer
for param in model.features.parameters():
    param.requires_grad = False
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]
features.extend([torch.nn.Linear(num_features, len(classes))])
model.classifier = torch.nn.Sequential(*features)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=decay_rate)

sys.stdout = open(os.path.join(model_dir, 'output.txt'), 'w')

print(model)

alexnet_trainer = trainer(model, optimizer, loss, scheduler, classes, train_set, test_set, model_dir)
alexnet_trainer.load_data(batch_size)
print("Start Training AlexNet")
alexnet_trainer.train(epochs)
print('Training time: {:0.1f} seconds.'.format(alexnet_trainer.train_time))
alexnet_trainer.save_model()
alexnet_trainer.generate_graphs()

print("End Training AlexNet")

message = """\
Subject: Training completed

Training AlexNet model completed.
"""

# send_email(message)

# ##############################################################
# #             ALEXNET WITH WEIGHT                            #
# ##############################################################
model_dir = '../models/alexnet_weight'

model = models.alexnet(pretrained=True)

# Transfer learning: Freeze all convolutional layers and only train the last liner layer of classifier layer
for param in model.features.parameters():
    param.requires_grad = False
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]
features.extend([torch.nn.Linear(num_features, len(classes))])
model.classifier = torch.nn.Sequential(*features)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = torch.nn.CrossEntropyLoss(weight=class_weight_ts)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=decay_rate)

sys.stdout = open(os.path.join(model_dir, 'output.txt'), 'w')

print(model)

alexnet_trainer = trainer(model, optimizer, loss, scheduler, classes, train_set, test_set, model_dir)
alexnet_trainer.load_data(batch_size)
print("Start Training AlexNet with weight ")
alexnet_trainer.train(epochs)
print('Training time: {:0.1f} seconds.'.format(alexnet_trainer.train_time))
alexnet_trainer.save_model()
alexnet_trainer.generate_graphs()

print("End Training AlexNet with weight ")

message = """\
Subject: Training completed

Training AlexNet with weight  model completed.
"""

# send_email(message)
