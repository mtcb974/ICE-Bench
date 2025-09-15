# Java Repository environment 

First, researchers need to download the Docker image we have prepared from this link: [url](TODO)

Next, use tmux to create a window and enter the docker container:
```bash
tmux new-session -s java-repo-env-1

docker run -it image_name bash
```

It is sufficient to keep the docker container running in the background.

