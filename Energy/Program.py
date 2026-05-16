from distutils import command
from operator import truediv
from textwrap import fill

import tkinter as tk
from turtle import width 
from tkinter import *
from tkinter import ttk
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from array import *
plt.style.use('fivethirtyeight')

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

#Load the MMU historical dataset
history = pd.read_excel('MMU Energy Consumption 2018-2021.xlsx')
df = pd.DataFrame(columns=['TimeStamp','Usage Peak (kwh)','Average pressure (Hg)','Average temperature','Average humidity (%)','Average wind speed (m/s)','Rainfall duration (min)','Rainfall amount (mm)','Type of day','Type of Lockdown'])
df2 = pd.DataFrame(columns=['TimeStamp','Usage Peak (kwh)'])

#Dataframe to save the predicted values
update_data = pd.DataFrame(columns=['TimeStamp','Predicted Usage Peak (kwh)'])
full_data = array('f',[])

def save_info():
    
    #Delete the previous dataset
    tv1.delete(*tv1.get_children())

    day_info = day.get()
    month_info = month.get()
    year_info = year.get()
    important_date = "1"
    tday_info =  tday.get()

    #Setup a new date format M/D/YYYY
    #Creating new algorithm
    #So if user prompt for type of day is 0, than it will take weekday energy consumption instead of weekend energy consumption

    if day_info<13 and tday_info==0:
        df2['Usage Peak (kwh)'] = history['Usage Peak (kwh)']
        df2['TimeStamp'] = pd.to_datetime(history['TimeStamp'])
        current_month = str(month_info)
        current_day = str(day_info+16)
        current_year = str(year_info-1)
        start_date = current_month+"/"+str(day_info+9)+"/"+current_year
        end_date = current_month+"/"+current_day+"/"+current_year
        mask = (df2['TimeStamp'] > start_date) & (df2['TimeStamp'] <= end_date)
        weekend_energy = df2.loc[mask]
        energy = weekend_energy['Usage Peak (kwh)'].max()
        string_date = str(day_info)+"/"+current_month+"/"+str(year_info)
        important_date = string_date

    elif day_info>20 and day_info<32 and tday_info==0:
        df2['Usage Peak (kwh)'] = history['Usage Peak (kwh)']
        df2['TimeStamp'] = pd.to_datetime(history['TimeStamp'])
        current_month = str(month_info)
        current_day = str(day_info)
        current_year = str(year_info-1)
        start_date = str(day_info-7)+"/"+current_month+"/"+current_year
        end_date = current_day+"/"+current_month+"/"+current_year  
        mask = (df2['TimeStamp'] > start_date) & (df2['TimeStamp'] <= end_date)
        weekend_energy = df2.loc[mask]
        energy = weekend_energy['Usage Peak (kwh)'].max() 
        string_date = current_day+"/"+current_month+"/"+str(year_info) 
        important_date = string_date

    elif day_info>12 and tday_info==0:

        df2['Usage Peak (kwh)'] = history['Usage Peak (kwh)']
        df2['TimeStamp'] = pd.to_datetime(history['TimeStamp'])
        current_month = str(month_info)
        current_day = str(day_info)
        current_year = str(year_info-1)
        start_date = current_day+"/"+current_month+"/"+current_year
        end_date = str(day_info+7)+"/"+current_month+"/"+current_year
        mask = (df2['TimeStamp'] > start_date) & (df2['TimeStamp'] <= end_date)
        weekend_energy = df2.loc[mask]
        energy = weekend_energy['Usage Peak (kwh)'].max() 
        string_date = current_day+"/"+current_month+"/"+str(year_info) 
        important_date = string_date

    elif day_info>20 and day_info<32 and tday_info==1:
        df2['Usage Peak (kwh)'] = history['Usage Peak (kwh)']
        df2['TimeStamp'] = pd.to_datetime(history['TimeStamp'])
        current_month = str(month_info)
        current_day = str(day_info)
        current_year = str(year_info-1)
        start_date = str(day_info-7)+"/"+current_month+"/"+current_year
        end_date = current_day+"/"+current_month+"/"+current_year
        mask = (df2['TimeStamp'] > start_date) & (df2['TimeStamp'] <= end_date)
        weekend_energy = df2.loc[mask]
        energy = weekend_energy['Usage Peak (kwh)'].min() 
        string_date = current_day+"/"+current_month+"/"+str(year_info) 
        important_date = string_date
        
    elif day_info>12 and tday_info==1:
        
        df2['Usage Peak (kwh)'] = history['Usage Peak (kwh)']
        df2['TimeStamp'] = pd.to_datetime(history['TimeStamp'])
        current_month = str(month_info)
        current_day = str(day_info)
        current_year = str(year_info-1)
        start_date = current_day+"/"+current_month+"/"+current_year
        end_date = str(day_info+7)+"/"+current_month+"/"+current_year
        mask = (df2['TimeStamp'] > start_date) & (df2['TimeStamp'] <= end_date)
        weekend_energy = df2.loc[mask]
        energy = weekend_energy['Usage Peak (kwh)'].min() 
        string_date = current_day+"/"+current_month+"/"+str(year_info) 
        important_date = string_date
        
    elif day_info<13 and tday_info==1:
        
        df2['Usage Peak (kwh)'] = history['Usage Peak (kwh)']
        df2['TimeStamp'] = pd.to_datetime(history['TimeStamp'])
        current_month = str(month_info)
        current_day = str(day_info+16)
        current_year = str(year_info-1)
        start_date = current_month+"/"+str(day_info+9)+"/"+current_year
        end_date = current_month+"/"+current_day+"/"+current_year
        mask = (df2['TimeStamp'] > start_date) & (df2['TimeStamp'] <= end_date)
        weekend_energy = df2.loc[mask]
        energy = weekend_energy['Usage Peak (kwh)'].min()
        string_date = str(day_info)+"/"+current_month+"/"+str(year_info)
        important_date = string_date

    tlockdown_info = tlockdown.get()
    temperature_info = float(temperature.get())
    humidity_info =  float(humidity.get())
    pressure_info = float(pressure.get())
    rainfallduration_info = float(rainfallduration.get())
    rainfallamount_info =  float(rainfallamount.get())
    windspeed_info = float(windspeed.get())
    
    #Store input data from user into a dataframe
    df.loc[df.shape[0]] = [important_date,energy,pressure_info,temperature_info,humidity_info,windspeed_info,rainfallduration_info,rainfallamount_info,tday_info, tlockdown_info]

    #Provide textline if the data input from the user is successfully updated into the database
    command_text.config(text = "Database is succesfully updated")

    #Update table with stored input data
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"

    for column in tv1["columns"]:
        tv1.heading(column,text=column)

    df_rows = df.to_numpy().tolist()

    for row in df_rows:
        tv1.insert("","end",values=row)
    return None

def update():
    tv2.delete(*tv2.get_children())

    update_data['Predicted Usage Peak (kwh)'] = full_data
    update_data['TimeStamp'] = df['TimeStamp']
    tv2["column"] = list(update_data.columns)
    tv2["show"] = "headings"

    for column in tv2["columns"]:
        tv2.heading(column,text=column)

    update_data_rows = update_data.to_numpy().tolist()

    for row in update_data_rows:
        tv2.insert("","end",values=row)
    return None

def train():
    #Load the MMU historical dataset
    history = pd.read_excel('MMU Energy Consumption 2018-2021.xlsx')
    dataset = history.append(df)
    print(dataset.tail())
    index = history.index
    number_of_rows = len(index)-1
    max_row = len(index)

    #Drop timestamp from the datasets
    values = dataset[['Usage Peak (kwh)','Average pressure (Hg)','Average temperature','Average humidity (%)','Average wind speed (m/s)','Rainfall duration (min)','Rainfall amount (mm)','Type of day','Type of Lockdown']].values 
      
    #Normalized the dataset values into the range of 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 1, 1)
   
    values = reframed.values

    #Split the data into train and validation datasets
    #All of the MMU historical energy consumption are used as training dataset
    n_train_time = number_of_rows
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    #Reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 
    # Reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].

    #Determine the model of LSTM
    #Set the hyperparameters of the LSTM, namely (number of epochs, batch size, number of neurons and learning rate)
    model = Sequential()
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])

    history = model.fit(train_X, train_y, epochs=2500, batch_size=70, validation_data=(test_X, test_y), verbose=1, shuffle=False)

    score = model.evaluate(test_X, test_y, batch_size=70, verbose=1)
    print('Test accuracy:', score[1])
    print('Accuracy:',score[1]*100)

    # Make a prediction by using the traing model
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0],17))
    # Invert value due to normalizations to obtain the correct value
    inv_yhat = np.concatenate((yhat, test_X[:, -8:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -8:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    # RMSE and MAE calculation
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mae = mean_absolute_error(inv_y, inv_yhat)
    print('Test MAE: %.3f' % mae)

    arr_actual = array('f',[])
    arr_forecast = array('f',[])

    for i in range(len(inv_y)):
        number = max_row+i 
        x =  inv_y[i] - inv_yhat[i]
        actual = dataset.iloc[number]['Usage Peak (kwh)']
        forecast = abs(dataset.iloc[number]['Usage Peak (kwh)'] + x)
        arr_actual.append(actual)
        arr_forecast.append(forecast)

    #Plot the graph using the predicted data
    global full_data 
    full_data = arr_forecast
    plt.plot(dataset["TimeStamp"][max_row:max_row+len(inv_y)], arr_forecast[:len(inv_y)], 'r', label="Prediction")
    plt.ylabel('Energy usage (kwh)', size=15)
    plt.xlabel('Date',fontsize=15,)
    plt.legend(fontsize=6)
    plt.show()

def clear_data():
    tv1.delete(*tv1.get_children())
    df.drop(df.index, inplace=True)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# Input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# Forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# Compile the data together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# Drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
    
#Development of GUI
screen = tk.Tk()
screen.geometry("870x920")
screen.title("MMU Energy Prediction")

command_text = Label(text= "",)
command_text.place(x=370,y=280)

date_text = Label(text= "Date : ",)
tlockdown_text = Label(text= "Type of lockdown * ",)
day_text = Label(text= "Day (DD)* ",)
month_text = Label(text= "Month (MM)* ",)
year_text = Label(text= "Year (YYYY)* ",)
temperature_text = Label(text= "Temperature (°C) * ",)
humidity_text = Label(text = "Relative Humidity (%) * ",)
pressure_text = Label(text= "Pressure (hPa) * ",)
rainfallduration_text = Label(text= "Rainfall Duration (min) * ",)
rainfallamount_text = Label(text= "Rainfall amount (mm) * ",)
windspeed_text = Label(text= "Wind speed (m/s) * ",)
tday_text = Label(text= "Type of day * ",)

date_text.place(x=150,y=30)
tlockdown_text.place(x=40,y=70)
day_text.place(x=270,y=0)
month_text.place(x=475,y=0)
year_text.place(x=705,y=0)
temperature_text.place(x=250,y=70)
humidity_text.place(x=450,y=70)
pressure_text.place(x=700,y=70)
rainfallduration_text.place(x=35,y=150)
rainfallamount_text.place(x=230,y=150)
windspeed_text.place(x=460,y=150)
tday_text.place(x=700,y=150)

#date = StringVar()
day = IntVar()
tlockdown = IntVar()
month = IntVar()
year = IntVar()
temperature = StringVar()
humidity = StringVar()
pressure = StringVar()
rainfallduration = StringVar()
rainfallamount = StringVar()
windspeed = StringVar()
tday = IntVar()

#date_entry = Entry(screen, textvariable= date, width= "20")
day_entry = Entry(screen, textvariable= day, width= "20")
month_entry = Entry(screen, textvariable= month, width= "20")
year_entry = Entry(screen, textvariable= year, width= "20")
tlockdown_entry = Entry(screen, textvariable= tlockdown, width= "20")
temperature_entry = Entry(screen,textvariable= temperature, width= "20")
humidity_entry = Entry(screen,textvariable= humidity, width= "20")
pressure_entry = Entry(screen,textvariable= pressure, width= "20")
rainfallduration_entry = Entry(screen,textvariable= rainfallduration, width= "20")
rainfallamount_entry = Entry(screen,textvariable= rainfallamount, width= "20")
windspeed_entry = Entry(screen,textvariable= windspeed, width= "20")
tday_entry = Entry(screen,textvariable= tday, width= "20")
#date_entry.place(x=35,y=100)
day_entry.place(x=235,y=30)
tlockdown_entry.place(x=35,y=100)
month_entry.place(x=450,y=30)
year_entry.place(x=675,y=30)
temperature_entry.place(x=235,y=100)
humidity_entry.place(x=450,y=100)
pressure_entry.place(x=675,y=100)
rainfallduration_entry.place(x=35,y=180)
rainfallamount_entry.place(x=230,y=180)
windspeed_entry.place(x=450,y=180)
tday_entry.place(x=675,y=180)

enter = Button(screen, text = "Enter", width="20", height= "2", command= save_info, bg = "grey")
enter.place(x=300,y=230)
clear = Button(screen, text = "Clear", width="20", height= "2", command= clear_data, bg = "grey")
clear.place(x=480,y=230)
clear = Button(screen, text = "Train", width="20", height= "2", command= train, bg = "grey")
clear.place(x=300,y=580)
clear = Button(screen, text = "Update", width="20", height= "2", command= update, bg = "grey")
clear.place(x=480,y=580)

#Frame for TreeView 0
frame1 = tk.LabelFrame(screen, text="Data")
frame1.place(x=35,y=310, height = 250, width= 800)

#Treeview widget 0
tv1 = ttk.Treeview(frame1)
tv1.place(relheight=1,relwidth=1)

treescolly = tk.Scrollbar(frame1, orient="vertical", command=tv1.yview)
treescollx = tk.Scrollbar(frame1, orient="horizontal", command=tv1.xview)
tv1.configure(xscrollcommand=treescollx.set,yscrollcommand=treescolly.set)
treescollx.pack(side = "bottom", fill="x")
treescolly.pack(side="right", fill="y")

#Frame for TreeView 1
frame2 = tk.LabelFrame(screen, text="Predicted Data")
frame2.place(x=35,y=650, height = 250, width= 800)

#Treeview widget 0
tv2 = ttk.Treeview(frame2)
tv2.place(relheight=1,relwidth=1)

treescolly2 = tk.Scrollbar(frame2, orient="vertical", command=tv2.yview)
treescollx2 = tk.Scrollbar(frame2, orient="horizontal", command=tv2.xview)
tv2.configure(xscrollcommand=treescollx2.set,yscrollcommand=treescolly2.set)
treescollx2.pack(side = "bottom", fill="x")
treescolly2.pack(side="right", fill="y")
screen.mainloop()