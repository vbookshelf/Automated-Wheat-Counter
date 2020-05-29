## Automated-Wheat-Counter
A flask web app that automatically counts wheat heads on wheat photos.

[ Under Construction ]

The demo app will be live only until end June 2020.<br>
Demo App: 

<br>

<img src="http://wheatcounter.test.woza.work/assets/app_pic3.png" width="350"></img>

<br>

My goal for this project was to build and deploy a flask web app that can automatically count wheat heads on wheat photos. A user is able to submit a photo and get an instant prediction.

This app could help researchers get quick rough estimates of wheat density.

#### Validation Results

Validation set percentage error: 23.03 %

Validation set MAE: 5.65


-x-

The process used to build and train the model is described in this Kaggle notebook:<br>


The model was fine tuned using data made available during the Kaggle Global Wheat Detection competition. The data was released under an open source MIT license.<br>
https://www.kaggle.com/c/global-wheat-detection

#### Server Deployment

The frontend and backend code is available in this repo. The models were too large to be uploaded. Please download them from the Kaggle notebook. There are two models: seg_model.pt and reg_model.pt. Please put both models into the folder called 'flask' before uploading the folder called 'wheat-backend' to your server.



The code is set up to be run as a Docker container. The code is based on this video tutorial:

Julian Nash docker and flask video tutorial<br>
https://www.youtube.com/watch?v=dVEjSmKFUVI


The .dockerignore file may not be visible. Please create this file if you don't see it. I suggest that you deploy on a Linux server running Ubuntu 16.04. In this repo I've included a pdf that explains the steps for installing Docker and Docker Compose on a Linux server. There is folder called 'static' containing a predict.html file. This folder is not essential and can be deleted. 
