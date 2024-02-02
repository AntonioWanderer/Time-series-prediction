import pandas
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input
from tensorflow.keras.models import Sequential

def graphical(x,y,name, newone):
    axs.cla()
    axs.plot(x, y)
    axs.scatter(x[-1]+1,newone)
    plt.grid(True)
    axs.set_title(name)
    axs.set_xlabel('year')
    axs.set_ylabel('Population')
    plt.savefig("results/" + name + ".jpg")

if __name__ == '__main__':
    df=pandas.read_csv("new.csv")
    fig, axs = plt.subplots(1, 1)
    for idx, row in df.iterrows():
        x = row.keys().tolist()
        y=row.tolist()
        name = y[0]
        x = list(map(float,x[4:-1]))
        y = list(map(float, y[4:-1]))

        cats = 1000
        x_train = np.array(x)
        y_train = np.array(y)

        start = min(y_train)-1
        stop = max(y_train)
        dy = (stop-start)/cats
        y_train_cat = np.zeros((len(y_train),cats))
        for j in range(len(y_train)):
            for i in range(cats):
                if start+i*dy<y_train[j]<=start+(i+1)*dy:
                    y_train_cat[j,i] = 1

        level = 10
        in_parts = np.array([y_train_cat[i:i+level] for i in range(len(y)-level)])
        out_parts = y_train[level:]
        value_from = min(out_parts)
        out_parts -= value_from
        value_to = max(out_parts)
        out_parts /= value_to
        #print(in_parts.shape, out_parts.shape)
        model = Sequential()
        model.add(Input(shape=(level,cats)))
        model.add(LSTM(4,activation="relu"))
        model.add(Dense(cats, activation="relu"))
        model.add(Dense(1, activation="relu"))

        print(model.summary())

        model.compile(loss="mse", metrics="mse")

        model.fit(in_parts,out_parts, epochs=100, batch_size=5)

        testing = y_train_cat[-level:]
        testing=np.expand_dims(testing,axis=0)
        newone = model.predict(testing)
        newone = value_from + (newone[0,0] * value_to)
        print(newone)

        graphical(x,y,name, newone)