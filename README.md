# TDS Project 1

The AI API provided by the university is used.
The API_TOKEN is detected from the environment variable assigned when running the docker container.
<br>
<br>
To pull the docker image:

```
sudo docker pull quasarleaf/my-fastapi-app:latest
```


To run the container:

```
sudo docker run -p 8000:8000 -e API_TOKEN=$API_TOKEN quasarleaf/my-fastapi-app:latest
```

Where API_TOKEN is the api token of the AI.

Example request to the project:

```
curl -X 'POST' "http://127.0.0.1:8000/run?task=detect%20text%20from%20image" -H "accept: application/json"
```

