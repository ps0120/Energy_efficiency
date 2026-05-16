from distutils import command
from operator import truediv
from textwrap import fill

import tkinter as tk
from turtle import width 
from tkinter import *
from tkinter import ttk
import pandas as pd
import calendar
import numpy as np
import matplotlib.pyplot as plt
from array import *
plt.style.use('fivethirtyeight')

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

#Load the MMU historical dataset
history = pd.read_excel('MMU Energy Consumption 2018-2021.xlsx', dtype={'TimeStamp': str})

#replace MMU Energy Consumption date data
def fix_date_format(date_str):

    if pd.isna(date_str):
        return pd.NaT
    
    date_str = str(date_str).split(' ')[0]
    parts = date_str.replace('-', '/').split('/')
    
    if len(parts) != 3:
        return pd.NaT
    
    try:
        first = int(float(parts[0]))
        second = int(float(parts[1]))
        year = int(float(parts[2]))
        
        if year < 100:
            year = year + 2000
        
        # if first > 12，then DD/MM/YYYY
        # if first <= 12，then MM/DD/YYYY
        if first > 12:
            day = first
            month = second
        else:
            month = first
            day = second
        
        return pd.Timestamp(year=year, month=month, day=day)
    except:
        return pd.NaT

# apply fix_date_format to the TimeStamp column
history['TimeStamp'] = history['TimeStamp'].apply(fix_date_format)

df = pd.DataFrame(columns=['TimeStamp','Usage Peak (kwh)','Average pressure (Hg)','Average temperature','Average humidity (%)','Average wind speed (m/s)','Rainfall duration (min)','Rainfall amount (mm)','Type of day','Type of Lockdown'])
df2 = pd.DataFrame(columns=['TimeStamp','Usage Peak (kwh)'])

#Dataframe to save the predicted values
update_data = pd.DataFrame(columns=['TimeStamp','Predicted Usage Peak (kwh)'])
full_data = array('f',[])

def save_info():
    tv1.delete(*tv1.get_children())

    try:
        day_info = day.get()
        month_info = month.get()
        year_info = year.get()
        tday_info = tday.get()
        tlockdown_info = tlockdown.get()
        
        # check valid date
        if day_info < 1 or day_info > 31 or month_info < 1 or month_info > 12:
            command_text.config(text="Invalid date entered")
            return None
        
        # check valid year: must be between 2018 and 2022
        if year_info < 2018 or year_info > 2022:
            command_text.config(text="Invalid year, must be between 2018-2022")
            return None
        
        # if user inputs 2022, use 2021 data instead
        query_year = year_info
        if year_info == 2022:
            query_year = 2021
        
        # filter data based on type of lockdown and type of day   
        df2 = pd.DataFrame()
        df2['Usage Peak (kwh)'] = history['Usage Peak (kwh)']
        df2['TimeStamp'] = history['TimeStamp'] 
        df2['Type of day'] = history['Type of day']
        df2['Type of Lockdown'] = history['Type of Lockdown']
        df2['Month'] = df2['TimeStamp'].dt.month
        df2['Year'] = df2['TimeStamp'].dt.year 
        df2['Day'] = df2['TimeStamp'].dt.day
        df2['OriginalIndex'] = history.index  #store the filtered data
        
        lockdown_mask = df2['Type of Lockdown'] == tlockdown_info
        day_type_mask = df2['Type of day'] == tday_info
        
        filtered_by_type = df2[lockdown_mask & day_type_mask]
        filtered_indices = set(filtered_by_type['OriginalIndex'].tolist())  #store the filtered data
        
        history_with_date = history.copy()
        history_with_date['Month'] = history_with_date['TimeStamp'].dt.month
        history_with_date['Year'] = history_with_date['TimeStamp'].dt.year
        history_with_date['Day'] = history_with_date['TimeStamp'].dt.day
        
        # find 13th day of the month and year based on user input
        day_13_idx_mask = (history_with_date['Month'] == month_info) & \
                          (history_with_date['Year'] == query_year) & \
                          (history_with_date['Day'] == 13)
        day_13_indices = history_with_date[day_13_idx_mask].index.tolist()
        
        
        
        if len(day_13_indices) > 0:
            day_13_row = day_13_indices[0]  
            
            # find 12 rows data (day 1-12)before 13th
            start_row = max(0, day_13_row - 12)
            before_13_indices = list(range(start_row, day_13_row))
            
            # find out of these 12 rows, which are in the filtered data from step 1
            before_13_filtered_indices = [idx for idx in before_13_indices if idx in filtered_indices]
           # before_13_filtered_data = history.loc[before_13_filtered_indices]
            
            
            days_in_month = calendar.monthrange(query_year, month_info)[1]
            
            # find all indices from 13th to end of month
            after_13_mask = (history_with_date['Month'] == month_info) & \
                           (history_with_date['Year'] == query_year) & \
                           (history_with_date['Day'] >= 13) & \
                           (history_with_date['Day'] <= days_in_month)
            after_13_all_indices = history_with_date[after_13_mask].index.tolist()
            
            # find out of these from 13th to end of month which are in the filtered data from step 1
            after_13_filtered_indices = [idx for idx in after_13_all_indices if idx in filtered_indices]
            
            
            all_filtered_indices = before_13_filtered_indices + after_13_filtered_indices
            combined_filtered_data = history.loc[all_filtered_indices]
            
            if len(combined_filtered_data) > 0:
                energy = combined_filtered_data['Usage Peak (kwh)'].mean()

            elif len(filtered_by_type) > 0:
                energy = filtered_by_type['Usage Peak (kwh)'].max()
                
        else:
            if len(filtered_by_type) > 0:
                energy = filtered_by_type['Usage Peak (kwh)'].max()
        
        
        string_date = f"{day_info}/{month_info}/{year_info}"
        important_date = string_date
        
    except Exception as e:
        command_text.config(text=f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

    temperature_info = float(temperature.get())
    humidity_info = float(humidity.get())
    pressure_info = float(pressure.get())
    rainfallduration_info = float(rainfallduration.get())
    rainfallamount_info = float(rainfallamount.get())
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
    
    #Convert TimeStamp to unified date format
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], dayfirst=True)
    df['TimeStamp'] = df['TimeStamp'].dt.strftime('%Y-%m-%d')
    
    dataset = pd.concat([history, df], ignore_index=True)
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
    n_train_time = number_of_rows
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]

    # 9 features: var1(t-1)~var9(t-1) are columns 0-8, var1(t)~var9(t) are columns 9-17
    n_features = 9
    train_X, train_y = train[:, :-n_features], train[:, 9]
    test_X, test_y = test[:, :-n_features], test[:, 9]
    
    # SVR doesn't need 3D input like LSTM, so we keep 2D shape
    # train_X shape: (samples, features)
    # No need to reshape for SVR
    
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 

    # Support Vector Regression (SVR) model
    model = SVR(kernel='rbf', C=100, epsilon=0.1)
    
    # Train the SVR model
    print("Training Support Vector Regression (SVR) model...")
    model.fit(train_X, train_y)
    print("SVR model training complete.")

    # Evaluate the SVR model
    score = model.score(test_X, test_y)
    print(f'SVR Test Score (R²): {score:.6f}')
    
    # Make a prediction by using the SVR model
    yhat = model.predict(test_X)
    
    # Reshape yhat to 2D for inverse transform compatibility
    yhat = yhat.reshape(-1, 1)

    # reverse scaling for forecasted
    inv_yhat_full = np.zeros((len(yhat), n_features))
    inv_yhat_full[:, 0] = yhat[:, 0]
    inv_yhat = scaler.inverse_transform(inv_yhat_full)
    inv_yhat = inv_yhat[:, 0]

    # reverse scaling for actual
    inv_y_full = np.zeros((len(test_y), n_features))
    inv_y_full[:, 0] = test_y
    inv_y = scaler.inverse_transform(inv_y_full)
    inv_y = inv_y[:, 0]

    print("\n" + "="*60)
    print("PREDICTION DETAILS")
    print("="*60)
    for i in range(len(inv_y)):
        print(f"Sample {i+1}: inv_y={inv_y[i]:.5f} kWh, inv_yhat={inv_yhat[i]:.5f} kWh, diff={inv_y[i]-inv_yhat[i]:.5f}")
    print("="*60 + "\n")
    
    inv_yhat_median = np.median(inv_yhat)
    inv_yhat_std = np.std(inv_yhat)
    
    for i in range(len(inv_yhat)):
        if abs(inv_yhat[i] - inv_yhat_median) > 2 * inv_yhat_std:
            if i == 0 and len(inv_yhat) > 1:
                inv_yhat[i] = inv_yhat[i+1]
            elif i > 0:
                inv_yhat[i] = (inv_yhat[i-1] + inv_yhat[i]) / 2
    

    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mae = mean_absolute_error(inv_y, inv_yhat)
    print('Test MAE: %.3f' % mae)

    mape = np.mean(np.abs((inv_y - inv_yhat) / inv_y)) * 100
    print('Test MAPE: %.2f%%' % mape)
    
    from sklearn.metrics import r2_score
    r2 = r2_score(inv_y, inv_yhat)
    print('Test R² Score: %.6f' % r2)
    
    rmse_percentage = (rmse / np.mean(inv_y)) * 100
    print('RMSE Percentage: %.2f%%' % rmse_percentage)
    
    from sklearn.metrics import median_absolute_error
    mdae = median_absolute_error(inv_y, inv_yhat)
    print('Median Absolute Error: %.3f' % mdae)

    arr_actual = array('f',[])
    arr_forecast = array('f',[])

    for i in range(len(inv_y)):
        number = max_row+i 
        x = inv_y[i] - inv_yhat[i]
        actual = dataset.iloc[number]['Usage Peak (kwh)']
        forecast = abs(dataset.iloc[number]['Usage Peak (kwh)'] + x)
        arr_actual.append(actual)
        arr_forecast.append(forecast)

    global full_data 
    full_data = arr_forecast
    
    plt.figure(figsize=(12, 6))
    plt.plot(dataset["TimeStamp"][max_row:max_row+len(inv_y)], arr_forecast[:len(inv_y)], 'r-', label="Prediction", linewidth=2)
    plt.ylabel('Energy usage (kwh)', size=15)
    plt.xlabel('Date', fontsize=15)
    plt.legend(fontsize=10)
    plt.tight_layout()
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
screen.title("MMU Energy Prediction (SVR)")

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