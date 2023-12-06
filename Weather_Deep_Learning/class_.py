import matplotlib.pyplot as plt
import requests
import torch
import numpy as np
import torch.nn as nn
import csv
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import os 

class Weather():

    def response(lat,long):
        today=date.today()
        day=today.strftime("%d")
        month=today.strftime("%m")
        new_date=str(today.year)+"-"+str(month)+"-"+str(day)
        old_date=str(today.year-30)+"-"+str(month)+"-"+str(day)
        api_url=f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={long}&start_date={old_date}&end_date={new_date}&daily=temperature_2m_mean,precipitation_sum,rain_sum'
        #api_url = 'https://archive-api.open-meteo.com/v1/archive?latitude=40.422&longitude=-3.695&start_date=1995-11-16&end_date=2023-11-30&daily=temperature_2m_mean,precipitation_sum,rain_sum'
        response=requests.get(api_url)
        resp_lan=response.json()

        return resp_lan
    
    def data_split(resp_lan):
        time=resp_lan['daily']['time']
        temp_mean=resp_lan['daily']['temperature_2m_mean']
        prec_sum=resp_lan['daily']['precipitation_sum']
        rain_sum=resp_lan['daily']['rain_sum']
        return time,temp_mean, prec_sum, rain_sum

    
    def plot(time,temp_mean,prec_sum,rain_sum):  #Create a graph with multiple variables

        plt.figure(figsize=(15,10),dpi=100)
        plt.plot(time,temp_mean,lw=1,color='red',linestyle='-',label='Mean temperature (Celsius)')
        plt.plot(time,prec_sum,lw=1,color='green',linestyle='-',label='precipitation sum (mm)')
        plt.plot(time,rain_sum,lw=1,color='blue',linestyle='-',label='rain sum (mm)')
        plt.legend(loc='upper center')
        plt.title(f"Weather condition for the last 30 years",fontsize=10)
        plt.xlabel('Time in days')
        plt.ylabel('Weather output')
        plt.xlim(time[0],time[-1])
        plt.grid()
        plt.show()

    def create_csv(time,temp_mean,prec_sum,rain_sum):
        with open('weather.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            field = [ "number","mean temp", "precipitation sum", "rain sum"]
            writer.writerow(field)

            for i in range (0,len(temp_mean)-2,1):
                writer.writerow([i+1, temp_mean[i], prec_sum[i],rain_sum[i]])

class CNN_LSTM(nn.Module):
    def __init__(self, conv_input,input_size, hidden_size, num_layers, output_size):
        super(CNN_LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv=nn.Conv1d(conv_input,conv_input,1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x=self.conv(x)
        h0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size)
        #print(f"x.shape:{x.shape},h0.shape:{h0.shape},c0.shape:{c0.shape}")
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
def split_data(data,time_step=12):
    dataX=[]
    datay=[]
    for i in range(len(data)-time_step):
        dataX.append(data[i:i+time_step])
        datay.append(data[i+time_step])
    dataX=np.array(dataX).reshape(len(dataX),time_step,-1)
    datay=np.array(datay)
    return dataX,datay

def train_test_split(dataX,datay,shuffle=True,percentage=0.8):
    if shuffle:
        random_num=[index for index in range(len(dataX))]
        np.random.shuffle(random_num)
        dataX=dataX[random_num]
        datay=datay[random_num]
    split_num=int(len(dataX)*percentage)
    train_X=dataX[:split_num]
    train_y=datay[:split_num]
    test_X=dataX[split_num:]
    test_y=datay[split_num:]
    return train_X,train_y,test_X,test_y

def mse(pred_y,true_y):
    return np.mean((pred_y-true_y) ** 2)

def MA_calculus(close):
    MA=[]
    sum=0
    n=0
    for i in close:
        sum+=i
        n+=1
        calculus=(sum/n)
        MA.append(calculus)

    return MA

def new_year(pred_y, dif):

    new_year=[]

    for i in range(0,365,1):
        new_year.append(pred_y[len(pred_y)-365+i]+dif)

    return new_year

def deep_learning(latitude,longitude):
    tiempo=Weather.response(latitude,longitude)
    roto=Weather.data_split(tiempo)
    fichero=Weather.create_csv(roto[0],roto[1],roto[2],roto[3])
    grafico=Weather.plot(roto[0],roto[1],roto[2],roto[3])
    train_df=pd.read_csv("weather.csv")

    fig,ax=plt.subplots(figsize=(12,8))
    plt.title("Mean temp for the last 30 years")
    ax.plot([i for i in range(len(train_df['mean temp']))],train_df['mean temp'])
    ax.set_xlabel("Time in days")
    ax.set_ylabel("Mean temperature in Celsius")
    plt.show()

    meantemp=train_df['mean temp'].values
    scaler=MinMaxScaler()
    meantemp=scaler.fit_transform(meantemp.reshape(-1,1))

    dataX,datay=split_data(meantemp,time_step=12)
    print(f"dataX.shape:{dataX.shape},datay.shape:{datay.shape}")

    train_X,train_y,test_X,test_y=train_test_split(dataX,datay,shuffle=False,percentage=0.8)
    print(f"train_X.shape:{train_X.shape},test_X.shape:{test_X.shape}")

    X_train,y_train=train_X,train_y

    test_X1=torch.Tensor(test_X)
    test_y1=torch.Tensor(test_y)


    input_size = 1
    conv_input=12
    hidden_size = 64
    num_layers = 5
    output_size = 1


    model =CNN_LSTM(conv_input,input_size, hidden_size, num_layers, output_size)


    num_epochs=500
    batch_size=64

    optimizer=optim.Adam(model.parameters(),lr=0.0001,betas=(0.5,0.999))

    criterion=nn.MSELoss()

    train_losses=[]
    test_losses=[]

    print(f"start")
    for epoch in range(num_epochs):

        random_num=[i for i in range(len(train_X))]
        np.random.shuffle(random_num)

        train_X=train_X[random_num]
        train_y=train_y[random_num]

        train_X1=torch.Tensor(train_X[:batch_size])
        train_y1=torch.Tensor(train_y[:batch_size])


        model.train()

        optimizer.zero_grad()

        output=model(train_X1)

        train_loss=criterion(output,train_y1)

        train_loss.backward()

        optimizer.step()

        if epoch%50==0:
            model.eval()
            with torch.no_grad():
                output=model(test_X1)
                test_loss=criterion(output,test_y1)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"epoch:{epoch},train_loss:{train_loss},test_loss:{test_loss}")

    train_X1=torch.Tensor(X_train)
    train_pred=model(train_X1).detach().numpy()
    test_pred=model(test_X1).detach().numpy()
    pred_y=np.concatenate((train_pred,test_pred))
    pred_y=scaler.inverse_transform(pred_y).T[0]
    true_y=np.concatenate((y_train,test_y))
    true_y=scaler.inverse_transform(true_y).T[0]
    print(f"mse(pred_y,true_y):{mse(pred_y,true_y)}")

    MA_calc=MA_calculus(pred_y)
    dif=MA_calc[len(MA_calc)-1]-MA_calc[len(MA_calc)-366]
    fig, ax=plt.subplots(2,1,figsize=(15,10))
    ax[0].set_title("CNN_LSTM")
    x=[i for i in range(len(true_y))]
    ax[0].plot(x,true_y,marker="x",markersize=1,label="true_y")
    ax[0].plot(x,pred_y,marker="o",markersize=1,label="pred_y")
    ax[0].set_xlabel("time in days")
    ax[0].set_ylabel("Mean temp in Celsius ")
    ax[0].legend()
    #plt.plot(x,true_y,marker="x",markersize=1,label="true_y")
    ax[1].set_title("Mean temp with median average")
    ax[1].plot(x,pred_y,marker="o",markersize=1,label="pred_y")
    ax[1].plot(x,MA_calc, markersize=1, label="median average")
    ax[1].set_xlabel("Time in days")
    ax[1].set_ylabel("Mean temp in Celsius")
    plt.legend()
    plt.show()



    year_pred=new_year(pred_y,dif)

    fig,ax=plt.subplots(figsize=(12,8))
    plt.title("Next year prediction")
    x=[i for i in range(len(year_pred))]
    ax.plot(x,year_pred,marker="o",markersize=1,label="pred_y")
    ax.set_xlabel("Time in days")
    ax.set_ylabel("Mean temperature in Celsius")
    plt.legend()
    plt.show()

    file = 'weather.csv'
    if(os.path.exists(file) and os.path.isfile(file)): 
        os.remove(file)
    else:
        pass

