FROM debian:latest

#  $ docker build . -t continuumio/anaconda3:latest -t continuumio/anaconda3:5.3.0
#  $ docker run --rm -it continuumio/anaconda3:latest /bin/bash
#  $ docker push continuumio/anaconda3:latest
#  $ docker push continuumio/anaconda3:5.3.0

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

RUN pip install --upgrade pip

#Prerequisites for Multinest

RUN apt-get update && apt-get install -y libblas3 libblas-dev liblapack3 liblapack-dev libatlas3-base libatlas-base-dev cmake build-essential git gfortran

RUN mkdir /samplers

RUN cd /samplers && git clone https://github.com/farhanferoz/MultiNest.git
RUN cd /samplers/MultiNest/MultiNest_v3.12_CMake/multinest && mkdir build && cd build \
&& cmake .. && make && cd

ENV LD_LIBRARY_PATH /samplers/MultiNest/MultiNest_v3.12_CMake/multinest/lib

# Install PyMultiNest
RUN pip install pymultinest

RUN pip install astroML colorednoise scaleogram george corner
RUN pip install scipy==1.7.1

RUN pip install matplotlib==3.1.0
