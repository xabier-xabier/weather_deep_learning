# weather_deep_learning
The aim of this app is to collect data about the weather for the last 30 years based on the user location input. 
For the location there are 2 options, there is a list from where the user can choose a place from a list of cities or 
the user can choose “custom” option which means that the user will have to write the coordinates (“Latitude” and “longitude”) of the desired place.

The app will collect data about the mean temperature, precipitation and rain sum for the chosen place.
Based on the mean temperature data the app will run a deep learning model to fit the data and get an approximate model for predictions.
Once the model is ready a graph will be plot with the model with the median average and a simulation for the next year weather based on the mean temperature data.
