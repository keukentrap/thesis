from malconv import Malconv
from preprocess import preprocess
import utils
import pandas as pd

fn = "input/small.csv"

model = Malconv()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

df = pd.read_csv(fn, header=None)
data, label = df[0].values, df[1].values
#data = preprocess(filenames)
x_train, x_test, y_train, y_test = utils.train_test_split(data, label)

history = model.fit(x_train, y_train)
pred = model.predict(x_test)
