import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
import time
import numpy as np
import os
# import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

test_transforms = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor()
    ])

test_data_path = 'scripts/regconition2/data'
test_data = torchvision.datasets.ImageFolder(root = test_data_path, transform = test_transforms)
test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False)

numClasses = 43

num = range(numClasses)
labels = []
for i in num:
    labels.append(str(i))
labels = sorted(labels)
for i in num:
    labels[i] = int(labels[i])
print("List of labels : ")
print("Actual labels \t--> Class in PyTorch")
# for i in num:
#     print("\t%d \t--> \t%d" % (labels[i], i))


# df = pd.read_csv("../data/Test.csv")
# numExamples = len(df)
# labels_list = list(df.ClassId)

from scripts.regconition2.class_alexnetTS import AlexnetTS
MODEL_PATH = "scripts/regconition2/traffic_models/pytorch_classification_alexnetTS.pth"
model = AlexnetTS(numClasses)
if torch.cuda.is_available():
    print("Run with cuda")
    print("\n")
# model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.load_state_dict(torch.load(MODEL_PATH))
model = model.cuda()

y_pred_list = []
corr_classified = 0

with torch.no_grad():
    model.eval()

    i = 0

    for image, _ in test_loader:
        image = image.cuda()

        y_test_pred = model(image)

        y_pred_softmax = torch.log_softmax(y_test_pred[0], dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        y_pred_tags = y_pred_tags.cpu().numpy()
        
        y_pred = y_pred_tags[0]
        y_pred = labels[y_pred]

        type_traff = ""
        if (y_pred ==0 or y_pred ==1 or y_pred ==2 or y_pred ==3 or y_pred ==4 or y_pred ==5 or y_pred ==6 \
        	or y_pred ==7 or y_pred ==8 or y_pred ==9 or y_pred ==10 or y_pred ==15 or y_pred ==16):
        	type_traff = "prohibitory"
        elif (y_pred ==33 or y_pred ==34 or y_pred ==35 or y_pred ==36 or y_pred ==37 or y_pred ==38 \
        	or y_pred ==39 or y_pred ==40):
        	type_traff = "mandatory"
        else:
        	type_traff = "danger"

        
        # y_pred_list.append(y_pred)
        y_pred_list.append(type_traff)

        # if labels_list[i] == y_pred:
        #     corr_classified += 1

        i += 1

# print("Running the model successfully")
# print("Number of correctly classified images = %d" % corr_classified)
# print("Number of incorrectly classified images = %d" % (numExamples - corr_classified))
# print("Final accuracy = %f" % (corr_classified / numExamples))

fig, axs = plt.subplots(2,4,figsize=(70,80))
fig.tight_layout(h_pad = 20)
# print("Running the model successfully")
for i in range(8):
    row = i // 4
    col = i % 4
    if i < 10:
    	imgName = 'scripts/regconition2/data/Test/0000'+str(i)+".png"
    else:
    	imgName = 'scripts/regconition2/data/Test/000'+str(i)+".png"
    img = Image.open(imgName)
    axs[row, col].imshow(img)
    # title = "Pred: %d, Actual: %d" % (y_pred_list[i], labels_list[i])
    title = "Pred: %s" % (y_pred_list[i])
    axs[row, col].set_title(title, fontsize=120)
# print("Running the model successfully")
plt.savefig("scripts/regconition2/predictions.png", bbox_inches = 'tight', pad_inches=0.5)