from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image

PATH_IMAGES = "../Datasets/scan_doc_rotation/images/"
PATH_LABELS = "../Datasets/scan_doc_rotation/labels/"
namesOfTrainImg = open("../Datasets/scan_doc_rotation/train_list.json", "r")
arrOfImages = []
arrOfLabels = []

N = 500
for name in namesOfTrainImg:
    if len(name) > 4:
        name = name.split()[0][1:-1]
        if name[-1] == '"':
            name = name[:-1]
        pathOnCurrImg = PATH_IMAGES + name
        img = Image.open(pathOnCurrImg)
        arrOfImages.append(img.getdata())
        fileOfLabel = open(PATH_LABELS + name[:-3] + "txt")
        label = float(fileOfLabel.read().splitlines()[0])
        # # label += 5
        # # label = label / 10
        # label = round(label * 2) / 2
        label = int(label)
        arrOfLabels.append(label)
        fileOfLabel.close()
    if len(arrOfLabels) == N:
        break

namesOfTrainImg.close()

allClasses = sorted(list(set(arrOfLabels)))
countOfClasses = len(allClasses)
print(countOfClasses, allClasses)
for i in range(len(arrOfLabels)):
    arrOfLabels[i] = allClasses.index(arrOfLabels[i])


train = np.array(arrOfImages)/255

trainTenzor = tf.constant(train, tf.float32)
trainLabels = tf.constant(arrOfLabels, tf.float32)
print(trainTenzor.shape)
print(trainLabels.shape)
print(trainLabels[:10])
# model = keras.Sequential([
#     # keras.layers.Dense(400, activation='relu'),
#     keras.layers.Dense(countOfClasses, activation='softmax')])

model = keras.Sequential([
    # keras.layers.Dense(500, activation='sigmoid'),
    keras.layers.Dense(countOfClasses, activation='softmax')])


model.compile(optimizer='adam',
              loss='SparseCategoricalCrossentropy',
              metrics=['MeanSquaredError', 'accuracy'])

N = len(trainTenzor) - len(trainTenzor)//5
print(N)
model.fit(trainTenzor[:N], trainLabels[:N], epochs=10, batch_size=32)

test, testLabels = trainTenzor[N:], trainLabels[N:]
test_loss, teseMse, test_acc = model.evaluate(test, testLabels)
print(test_loss, teseMse, test_acc)
# net = tf.keras.models.load_model("C:\\Users\\Norma\\PycharmProjects\\Machine Learning\\Experiments and smth like this\\DocTurn")
print(np.round(model.__call__(trainTenzor).numpy()[:2], 2))
res = np.round(model.__call__(test).numpy(), 2)
correct = 0
testLabels = np.int32(testLabels)
for i in range(len(res)):
    answer = res[i]
    errorInClass = 1
    predictWithLittleError = answer[testLabels[i] - errorInClass:testLabels[i] + errorInClass + 1].sum()
    #print(predictWithLittleError)
    if predictWithLittleError > 0.9:
        correct += 1
print(correct / len(res))
