# MentalQ - C242-PS246

## Table of Contents

1. [Description](#Description)
2. [Team](#C242-PS246---ml)
4. [API Endpoint](#Technology)

## Description
The MentalQ app utilizes **a robust cloud architecture** powered by **Google Cloud**. We’ve implemented a **CI/CD pipeline** that integrates our API from GitHub into Cloud Build, automatically packaging the app into a Dockerfile and storing it in **Artifact Registry**. The app is then deployed to two synchronized **Cloud Run** instances, **ensuring high availability**.

Our app services are powered by **Cloud Storage** and **Cloud SQL** for **efficient data management**. **Cloud IAM** and **Secret Manager** are **integrated** into the architecture to provide **secure environment** management, ensuring that sensitive information such as API keys and configurations are securely handled. This integration with Secret Manager ensures that our deployment pipeline meets the **highest standards of security** while providing **a seamless, automated process** from development to deployment.

**Once set up**, the API is ready to be **consumed** by the MentalQ app, **offering secure, scalable, and reliable mental health support to users.**


## C242-PS246 - ML

| Bangkit ID | Name | Learning Path | University | LinkedIn |
| ---      | ---       | ---       | ---       | ---       |
| M129B4KX2462 | Melinda Naurah Salsabila | Machine Learning | Politeknik Negeri Jember | [![text](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/melinda-naurah/) |
| M227B4KY3579 | Rafi Achmad Fahreza | Machine Learning | Universitas Jember | [![text](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rafiachmadfr/) |
| M129B4KY1504 | Fikri Faqih Al Fawwaz | Machine Learning | Politeknik Negeri Jember | [![text](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/fikrifaqihalfawwaz/) |

## Technology

- Flask
- TensorFlow and Keras
- NumPy
- More++

## Installation and Usage of Flask Application

This application uses Flask as a web framework to create an image prediction API using a TensorFlow model. Several dependencies must be installed before running the application.

## Requirement
Make sure you have installed:

Python 3.9 or newer
pip (package installer for Python)

## Installation Steps

1. Clone the Repository
Clone the repository from GitHub to your computer.

```bash
git clone https://github.com/MentalQ-App/MentalQ-ML.git
cd NONE
```

2. Ensure Model and Data Availability
Make sure the model file (model.h5) is available in the project's root directory.

3. Run the Application
Run the Flask application using the following command:

```bash
python local.py
```

Open your browser and go to http://127.0.0.1:8080 to check the API.

## API Endpoints

Predict
This endpoint is used to upload an image and get a prediction.

URL: /predict
Method: POST
Content-Type: multipart/form-data
Form Data:
image: File gambar (jpg, jpeg, png)

## Usage

Use the /predict endpoint to make predictions based on the input image.

```
curl -X POST -F "image=@path/to/your/image.jpg" http://127.0.0.1:8080/predict
```

Replace path/to/your/image.jpg with the path to the image file you want to predict.

## Example Response

```
{

}
```
