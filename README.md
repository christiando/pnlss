# PNLSS Code

This repo provides the necessary code for reproducing the results of the paper...

## Setup

For setting up the environment we assume that you have installed Docker on your machine. The repo comes with a _Dockerfile__, which creates an image with the necessary dependencies. So clone the repository and build the image.

```bash
git clone https://github.com/christiando/pnlss.git
cd pnlss
docker build -t pnlss .
```

This might take a couple of minutes. Once the image is build you can start a jupyter lab environment.

```bash
docker run -p 8888:8888 -v $(pwd):/app/work pnlss
```

Click on the link, that is prompted in your terminal, and you can execute the notebooks that come with this repository.
