import re
import PIL
import gym
import random
import numpy as np
import tesserocr
import tflearn
import gensim
from PIL import Image
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
import cv2
import imutils
import universe
import atexit
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from random import randint

#button object
class Button:
    x = 0
    y = 0
    vector = []

    def __init__(self, x, y, vector):
        self.x = x
        self.y = y
        self.vector = vector

    def __repr__(self):
        string =  self.x , ", " , self.y
        return string


#Hier beginnen we !
print("Reading in...")
modelNLP = None
modelNLP = gensim.models.KeyedVectors.load_word2vec_format('/media/axel/C654805754804BDD/Users/vulst/Downloads/GoogleNews-vectors-negative300.bin', binary=True, limit=1000000)
print("done with reading")


def preprocess(observation):
    if(observation[0] != None):
        buttonList = []
        observation = np.array(observation[0]['vision'])  # convert list to 3D-array
        img = observation[125:387, 9:270]  # convert to 210-50x160 input (geel ook al uitgefilterd) anders x = 75
        cv2.imwrite('SequenceBefore.png', img)
        image = Image.open('SequenceBefore.png')
        # convert to grayscale
        img = image.convert('L')
        img_np = np.array(img)
        img_np = (img_np > 100) * 255
        img = PIL.Image.fromarray(img_np.astype(np.uint8))
        img = img.resize((int(img.size[0] * 3.5), int(img.size[1] * 3.5)), PIL.Image.BILINEAR)

        with tesserocr.PyTessBaseAPI() as api:
            api.SetImage(img)
            boxes = api.GetComponentImages(tesserocr.RIL.TEXTLINE, True)
            # print('Found {} textline image components.'.format(len(boxes)))
            if(len(boxes) == 5):
                for i, (im, box, _, _) in enumerate(boxes):
                    # im is a PIL image object
                    # box is a dict with x, y, w and h keys
                    api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
                    ocrResult = api.GetUTF8Text().lower()
                    conf = api.MeanTextConf()

                    ocrResult = re.sub('[\s+]', '', ocrResult)
                    print("the word is: " , ocrResult)
                    #TODO vecotr afhankelijk van OCRresult
                    try:
                        vector = modelNLP.wv[ocrResult]
                    except KeyError as err:
                        vector = modelNLP.wv["house"]
                        print("word not found in google...")
                        # return None
                    x__ = box['x']/3.3 + 15
                    y__ = box['y']/3.3 + 140
                    # print("x: " , x__)
                    # print("y: " , y__)
                    buttonList.append(Button(x__, y__, vector))
        if len(buttonList) == 5:
            return buttonList
        else:
            return None
    return None

def plotter(x ,y):
    plt.plot(x,y)
    plt.show()

def getstate(buttonList):
    vectorList = []
    # dit geeft me een array van allemaal arrays met 1 element in
    # batch, vector, channel =1
    for vec in range(len(buttonList)):
        vectorList.append(buttonList[vec].vector)
    vectorList = np.concatenate(vectorList)
    # print(vectorList)
    vectorList = vectorList.reshape([1, 1500])  # dus 3d vorm, dit moet want een tensor mag niet echt 1d zijn!
    # print(vectorList)
    return vectorList


env = gym.make('wob.mini.ClickButtonSequence-v0')
env.configure(remotes=1, fps=1,
              vnc_driver='go',
              vnc_kwargs={'encoding': 'tight', 'compress_level': 0,
                          'fine_quality_level': 100, 'subsample_level': 0})
observation = env.reset()
#these are for the RL step
input_size = 300*5 #ik geef 5 buttons in
howfar =0
buttonList = []
# just the scores that met our threshold:
x__ = 100
y__ = 200
totalEpisode = 0
# atexit.register(plotter, x, y)
resume = False;
# model = neural_network_model(input_size)

tf.reset_default_graph
#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1, input_size],dtype=tf.float32)
# indie vorm verdeeld tussen...
with tf.name_scope("fully_connected"):
    W = tf.Variable(tf.random_uniform([1500,5],0,0.02))
    Qout = tf.matmul(inputs1,W)
    predict = tf.argmax(Qout,1)
    tf.summary.histogram("weigths", W)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,5],dtype=tf.float32)
with tf.name_scope("loss"):
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    tf.summary.scalar("loss", loss)
with tf.name_scope("train"):
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()
merged_summary = tf.summary.merge_all()
saver = tf.train.Saver()

# Set learning parameters
y = .99
e = 0.9
num_episode = 0
#create lists to contain total rewards and steps per episode
rewardList = []
saved = False

with tf.Session() as sess:
    sess.run(init)
    howfar = +1
    buttonList = None
    # just the scores that met our threshold:
    x__ = 100
    y__ = 200
    r = [1]
    r[0] = 0
    writer = tf.summary.FileWriter("/tmp/tensorflow/SelfMadeRL/1Layer-6")
    writer.add_graph(sess.graph)
    if (saved):
        print("read model in ")
        saver.restore(sess, "selfmadeRL.ckpt")
    else:
        sess.run(init)
    #Just to start!
    while (observation == [None]):
        # env.render()
        # zolang dat er geen observation is kunnen we niets doen dus proberen we dit maar...
        action = [universe.spaces.PointerEvent(x__, y__, 0), universe.spaces.PointerEvent(x__, y__, 1),
                  universe.spaces.PointerEvent(x__, y__, 0), universe.spaces.PointerEvent(200, 300, 0)]
        action = [action for ob in observation]
        observation, reward_n, done_n, info = env.step(action)
        # soms zit hij precies vast hiermee is dit opgelost
        stuck = 0
        # print("aan het proberen...")
    if (observation != [None]):
        while(buttonList == None):
            observation, reward_n, done_n, info = env.step(action)
            buttonList = preprocess(observation)  # ik krijg een buttonlist met 5 buttons terug of ofwel none
        state = getstate(buttonList)
    while True:
        while True:
            # env.render()
            stuck = stuck + 1
            rAll = 0
            while(stuck > 10 or buttonList == None):
                print("I was stuck, but it is solved...")
                x__ = 300
                y__ = 300
                action = [universe.spaces.PointerEvent(x__, y__, 0), universe.spaces.PointerEvent(x__, y__, 1),
                          universe.spaces.PointerEvent(x__, y__, 0), universe.spaces.PointerEvent(200, 300, 0)]
                action = [action for ob in observation]
                observation, reward_n, done_n, info = env.step(action)
                # env.render()
                stuck = 0;
                buttonList = preprocess(observation)  # ik krijg een buttonlist met 5 buttons terug of ofwel none
                # input("Buttonlist was none")
                if(buttonList != None):
                    state = getstate(buttonList)
            # print("buttonlist is not none so we get the state now")
            stuck = 0;
            d = False
            # buttonList = OCRTestStub()
            if(buttonList != None):
                # print("buttonlist to check if not none: " , buttonList)
                action, allQ = sess.run([predict, Qout], feed_dict={inputs1: state})
                #action[0] is heeft de grootste probabilitietieut naar haalt hem gwn uit een array
                action = action[0]
                # heel soms eens random iets doen
                if np.random.rand(1) < e:
                    print("Random action picked")
                    action = randint(0, 4)
                print("Action on previous state: ", action)
                # input("waiting to do action")
                x__ = buttonList[action].x
                y__ = buttonList[action].y
                xyaction = [universe.spaces.PointerEvent(x__, y__, 0), universe.spaces.PointerEvent(x__, y__, 1),
                          universe.spaces.PointerEvent(x__, y__, 0),  universe.spaces.PointerEvent(200, 300, 0)]
                xyaction = [xyaction for ob in observation]
                observation, r, d, _ = env.step(xyaction)
                x__ = 300
                y__ = 300
                waitaction = [universe.spaces.PointerEvent(x__, y__, 0), universe.spaces.PointerEvent(x__, y__, 0),
                              universe.spaces.PointerEvent(x__, y__, 0)]
                waitaction = [waitaction for ob in observation]
                observation, r, d, _ = env.step(waitaction)
                # env.render()
                print("reward1: " , r , " done: " , d)
                # input("Action is just done with above reward")
                print("after the action is the view now: ")
                buttonList = preprocess(observation)  # ik krijg een buttonlist met 5 buttons terug of ofwel none
                print("state above is the new state (+ it will be the state to do the next on)")
                # input("You can check the new state")
                if(buttonList != None):
                    state1 = getstate(buttonList)
                    # Obtain the Q' values by feeding the new state through our network
                    # ge wilt nu alle qvalues voor de volgende terug krijgen zodat je de grootste kan bepalen die heb j enodig voor die formule
                    #hier moet ik die observatie s1 meegeven in de echte dinges!!
                    Q1 = sess.run(Qout, feed_dict={inputs1: state1})

                    # Obtain maxQ' and set our target value for chosen action.
                    maxQ1 = np.max(Q1)
                    targetQ = allQ
                    targetQ[0, action] = r[0] + y * maxQ1
                    # Train our network using target and predicted Q values
                    # hier doe je de backpropagation je moet al die functies bovenaan in elkaar invullen endan zie jdat je deze 2 waarden nodig ehbt (placeholders)
                    _, W1 = sess.run([updateModel, W], feed_dict={inputs1: state, nextQ: targetQ})

                    rAll += r[0]
                    state = state1
                if(action == 3 or action == 4 and not d[0]):
                    observation, r, d, _ = env.step(xyaction)
                    print("reward for second time: " , r , " done: " , d)
                if d[0]:
                    # Reduce chance of random action as we train the model.
                    print("resetted with: ", r[0])
                    if(r[0] > 0):
                        e = 1. / ((num_episode / 300) + 1)
                        #start bij .9 --> hoe lang random..? bij 300 episodes ng amar de helft..!
                        # input("the episode is resetted")
                    #some more randomness to discover something
                    e = 1. / ((num_episode / 50) + 3)
                    env.reset()
                    # input("Done, Press Enter to continue...")
                    num_episode += 1
                    rewardList.append(r[0])
                    if num_episode % 3 == 0:
                        [s] = sess.run([merged_summary], feed_dict={inputs1: state, nextQ: targetQ})
                        #Om de 3 keer wegschrijven en tensorboard updaten
                        writer.add_summary(s, num_episode)
                    if(num_episode%10 == 0):
                        gem = sum(rewardList)/num_episode
                        print("This is episode: " , num_episode , "with a current reward of: " , gem  )
                        print(rAll)
                    break
                    save_path = saver.save(sess, "selfmadeRL.ckpt")