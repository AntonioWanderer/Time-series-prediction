import pandas
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, GRU, Input
from tensorflow.keras.models import Sequential

class country:
    def __init__(self, keys=None, vals=None, name='Noname'):
        self.keys = np.array(keys,dtype="float64")
        self.vals = np.array(vals,dtype="float64")
        self.name = name
        self.net_in = np.array([0])
        self.net_out = np.array([0])
        self.depth = 5
        self.cats = 100
        self.model = None
        self.predict_depth = 30
        self.bounds = None
        self.trend = None

    def trendLine(self, add = False):
        if not add:
            z = np.polyfit(self.keys, self.vals, 2)
            self.trend = np.poly1d(z)
            self.vals -= self.trend(self.keys)
        else:
            self.vals += self.trend(self.keys)
    def toVector(self,value,columns,bounds):
        vector = np.zeros(columns)
        dx = (bounds[1] - bounds[0]-0.01)/columns
        for i in range(columns):
            if dx*i-0.01<value-bounds[0]<=dx*(i+1):
                vector[i] = 1
        return vector
    def toCategories(self, sequence, columns, bounds):
        return np.array([self.toVector(val, columns, bounds) for val in sequence])
    def tokenize(self):
        self.bounds = (min(self.vals),max(self.vals))
        y_train_cat = self.toCategories(self.vals,self.cats,self.bounds)
        in_parts = np.array([y_train_cat[i:i + self.depth] for i in range(len(self.keys) - self.depth)])
        out_parts = self.vals[self.depth:].copy()
        out_parts -= self.bounds[0]
        out_parts /= self.bounds[1]
        self.net_in = in_parts
        self.net_out = out_parts

    def getmodel(self):
        model = Sequential()
        model.add(Input(shape=(self.depth, self.cats)))
        model.add(GRU(4, activation="relu"))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="mse", metrics="mse")
        self.model = model
    def modelSummary(self):
        print(self.model.summary())
    def train(self):
        self.model.fit(self.net_in, self.net_out, epochs=100, batch_size=5)
    def prediction(self):
        testing = self.toCategories(self.vals[-self.depth:],self.cats,self.bounds)
        testing = np.expand_dims(testing, axis=0)
        newone = self.model.predict(testing)
        newone = self.bounds[0] + (newone[0, 0] * self.bounds[1])
        self.keys = np.append(self.keys,self.keys[-1]+1)
        self.vals = np.append(self.vals,newone)

    def graphical(self):
        fig, axs = plt.subplots(1, 1)
        axs.cla()
        axs.plot(self.keys[:-self.predict_depth], self.vals[:-self.predict_depth], color="blue")
        axs.plot(self.keys[-self.predict_depth-1:], self.vals[-self.predict_depth-1:], color="red")
        plt.grid(True)
        axs.set_title(self.name)
        axs.set_xlabel('year')
        axs.set_ylabel('Population')
        plt.savefig("results/" + self.name + ".jpg")

    def __add__(self, other):
        child = country(name="Summary")
        child.keys = self.keys
        child.vals = self.vals+other.vals
        if np.isnan(child.vals[0]):
            return self
        return child

def data_load(filename, read_from_line = 4):
    countries = []
    df = pandas.read_csv(filename)
    for idx, row in df.iterrows():
        x = row.keys().tolist()
        y=row.tolist()
        name = y[0]
        x = list(map(float,x[read_from_line:-1]))
        y = list(map(float, y[read_from_line:-1]))
        countries.append(country(x,y,name))
    return countries

if __name__ == '__main__':
    items = data_load("new.csv")

    common = items[0]
    for i in items[1:]:
        common = common+i

    common.trendLine()
    common.tokenize()
    common.getmodel()
    common.modelSummary()
    common.train()
    for i in range(common.predict_depth):
        common.prediction()
    common.trendLine(add=True)
    common.graphical()

    for item in items:
        item.trendLine()
        item.tokenize()
        item.getmodel()
        item.modelSummary()
        item.train()
        for i in range(item.predict_depth):
            item.prediction()
        item.trendLine(add=True)
        item.graphical()