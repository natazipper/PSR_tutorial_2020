# PSR_tutorial_2020
Getting information from your data. Non-parametric and parametric analysis.

To clone the directory:

git clone https://github.com/natazipper/PSR_tutorial_2020.git

To build the docker image (will take some time):

cd PSR_tutorial_2020/docker/
docker build -t data_stat .

To run the docker image:

docker run --rm -p 8888:8888 -v /your_path_to/PSR_tutorial_2020/code/:/PSR_tutor -ti data_stat bash

Inside the docker:

cd PSR_tutor
jupyter notebook --ip 0.0.0.0 --allow-root
