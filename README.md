# Project Title

TWEETTRUTH - Explainable Fake News Detection system


# Contact information

Honggyo Suh | z5355159@ad.unsw.edu.au | Machine Learning Developer  
Hayley Yu | z5218663@ad.unsw.edu.au | Scrum Master  
Xingjian Chen | z5360719@ad.unsw.edu.au | Software engineer (Frontend)  
Rowena (Yiming) Si | z5360492@ad.unsw.edu.au | Software engineer (Frontend)  
En Chen | z5335039@ad.unsw.edu.au | Software engineer (Frontend)  


# Installation & API usages

To install the project, use below code.  
git clone https://github.com/unsw-cse-comp3900-9900-23T3/capstone-project-3900m13aquantumquokkas.git  

Please install all libraries using below command.  
pip install -r requirements.txt  

Our system requires pre-installation and activation of LIWC software for CLI access.  
The system can only run while LIWC software is running and open.  
Please download LIWC software using below link, and activate with key in LIWC_serial_no.txt.   
https://www.liwc.app/download/validate?key=LIWC22-B2C7D9F4-5C2E4C7B-839CF85A-5C4E6526  

Please note that LIWC software was purchased by our client Jiaojiao Jiang, and can be accessed for 90 days from 2024/09/29 to 2024/12/28  

Please note that currently our system uses GPT-3.5 turbo with OpenAI API, using key in backend/API_key.txt.   
50$ credit for this API was purchased by our client Jiaojiao Jiang, may not be available after use.


# File explanation

backend contains fake news detection systems, model, training/testing files, and other files required for system.  
backend.py is the file contains every component used for detection, will be called by server to generate explanation.  
backend_for_system_test.ipynb, data_pipeline.ipynb, raw_Google_API.ipynb contains training/testing of models.  
API_key.txt, columns_to_predict, columns_to_scale, common_in_fake.txt, common_in_true.txt, LIWC_average_fake.txt are files required for system.  
logistic_regression.pkl, standard_scaler.pkl are detection model and scaler used for system.  


# Running the application

1. Start server with python server.py or python3 server.py if in mac environment.  
2. Type http://localhost:8000/frontend/index.html on your browser to access the website, if you would like to access from other device, change the address with IP address running the server.py.  
3. Type the text on the blank, and click detect fake news button.  
4. The result will be shown shortly, execution time can vary depends on the length of input text, network status, and OpenAI API status.  


# Troubleshooting Guide

If you cannot generate LIWC analysis with CLI access, check the followings.

1. Check if the software is correctly installed.
2. Check if the software is activated with key.
3. Check if the software is running and open.
4. Check if the software access is expired.

If you cannot generate query or explanation with OpenAI API access, check the followings.

1. Check if OpenAI API is in normal condition on https://status.openai.com
2. OpenAI API may not reply until the timeout or reply with empty strings without specific reason, please wait a while and try run again.