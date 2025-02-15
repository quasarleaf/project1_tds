# TDS Project 1

The AI API provided by the university is used.
The AIPROXY_TOKEN is detected from the environment variable assigned when running the docker container.
<br>
<br>
To pull the docker image:

```
sudo docker pull quasarleaf/my-fastapi-app:latest
```


To run the container:

```
sudo docker run -p 8000:8000 -e AIPROXY_TOKEN=$AIPROXY_TOKEN quasarleaf/my-fastapi-app:latest
```

Where AIPROXY_TOKEN is the api token of the AI.

Example request to the project:

```
curl -X 'POST' "http://127.0.0.1:8000/run?task=detect%20text%20from%20image" -H "accept: application/json"
```



