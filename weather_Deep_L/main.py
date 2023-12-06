from class_ import  deep_learning
import PySimpleGUI as psg


#set the theme for the screen/window
psg.theme('DarkGrey2')

#define layout
layout=[[psg.Text('Choose the city you want to check the weather history record',size=(30, 2), font='Lucida',justification='left')],
        [psg.Combo(['Custom','Madrid','Paris','Hong Kong', 'Moscow','New York', 'Tokyo', 'Shanghai','London','Berlin','Jakarta'],default_value='Custom',key='place')],
        [psg.Button('SEARCH', font=('Times New Roman',12)),psg.Button('CANCEL', font=('Times New Roman',12))]]

#Define Window
win =psg.Window('Check the weather forecast based on your location',layout)
j=0

while True:  # Event Loop
    event, values = win.read()
    if event in (psg.WIN_CLOSED, 'CANCEL'):         # If Cancel or close window, the program will end
        break
        
    else:
        if values['place']=="Custom":
            win.close()
            layout1=[[psg.Text('Enter the latitude of your place ',size=(40, 1), font='Lucida',justification='left')],
            [psg.InputText("", key='latitude')],
            [psg.Text('Enter the longitude of your place ',size=(40, 1), font='Lucida',justification='left')],
            [psg.InputText("", key='longitude')],
            [psg.Button('SEARCH', font=('Times New Roman',12)),psg.Button('CANCEL', font=('Times New Roman',12))]],
            win1 =psg.Window('Check the weather forecast based on your location',layout1)

            while True:
                event,values=win1.read()
                while j==0:
                    if event in (psg.WIN_CLOSED, 'CANCEL'): 
                        break
                    try:
                        float(values['longitude']) and float(values['latitude'])
                        j=1
                    except:
                        win1.close()
                        psg.popup("The latitude and longitude fileds are not numbers",title="Error, latitude or longitude data error")
                        layout1=[[psg.Text('Enter the latitude of your place ',size=(40, 1), font='Lucida',justification='left')],
                        [psg.InputText("", key='latitude')],
                        [psg.Text('Enter the longitude of your place ',size=(40, 1), font='Lucida',justification='left')],
                        [psg.InputText("", key='longitude')],
                        [psg.Button('SEARCH', font=('Times New Roman',12)),psg.Button('CANCEL', font=('Times New Roman',12))]],
                        win1 =psg.Window('Error, latitude or longitude data error',layout1)
                        event,values=win1.read()
                
                if event in (psg.WIN_CLOSED, 'CANCEL'): 
                    break

                elif values['latitude']=="" or values['longitude']=="":
                    psg.popup("The latitude and longitude fileds are empty",title="Error, latitude or longitude data empty")
                
                else:
                    modelo=deep_learning(values['latitude'],values['longitude'])
                    
                win1.close()


        elif values['place']=="Madrid":
            latitude=float(40.4130)
            longitude=float(-3.6842)
            modelo=deep_learning(latitude, longitude)
            win.close()

        elif values['place']=="Paris":
            latitude=float(48.8553)
            longitude=float(2.2988)
            modelo=deep_learning(latitude, longitude)
            win.close()

        elif values['place']=="Hong Kong":
            latitude=float(22.3140 )
            longitude=float(114.1802)
            modelo=deep_learning(latitude, longitude)
            win.close()

        elif values['place']=="Moscow":
            latitude=float(55.7398)
            longitude=float(37.6214)
            modelo=deep_learning(latitude, longitude)
            win.close()

        elif values['place']=="New York":
            latitude=float(40.7363)
            longitude=float(-73.9852)
            modelo=deep_learning(latitude, longitude)
            win.close()

        elif values['place']=="Tokyo":
            latitude=float(35.6989)
            longitude=float(139.7772)
            modelo=deep_learning(latitude, longitude)
            win.close()

        elif values['place']=="Shanghai":
            latitude=float(31.2087)
            longitude=float(121.4563)
            modelo=deep_learning(latitude, longitude)
            win.close()

        elif values['place']=="London":
            latitude=float(51.4989)
            longitude=float(-0.1290)
            modelo=deep_learning(latitude, longitude)
            win.close()

        elif values['place']=="Berlin":
            latitude=float(52.5142)
            longitude=float(13.3971)
            modelo=deep_learning(latitude, longitude)
            win.close()

        elif values['place']=="Jakarta":
            latitude=float(-6.2150)
            longitude=float(106.8413)
            modelo=deep_learning(latitude, longitude)
            win.close()

            

win.close()