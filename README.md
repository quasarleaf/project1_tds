# TDS Project 1

The AI API provided by the university is used.
The AIPROXY_TOKEN is detected from the environment variable assigned when running the docker container.
<br>
<br>

To run the container:

```
podman run -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 docker.io/quasarleaf/my-fastapi-app:latest
```

Where AIPROXY_TOKEN is the api token of the AI.

Run a generate data query first to populate the data folder:

```
curl -X 'POST' "http://127.0.0.1:8000/run" --get --data-urlencode task="generate data" -H "accept: application/json" && echo
```

Example request to the project:

```
curl -X 'POST' "http://127.0.0.1:8000/run?task=detect%20text%20from%20image" -H "accept: application/json"
```



