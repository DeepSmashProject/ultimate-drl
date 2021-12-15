FROM deepsmash/yuzu_emu:latest

WORKDIR /workspace
RUN git clone https://github.com/DeepSmashProject/yuzulib.git
RUN git clone https://github.com/DeepSmashProject/libultimate.git
RUN git clone https://github.com/DeepSmashProject/ultimate-gym.git
RUN git clone https://github.com/DeepSmashProject/ultimate-drl.git
RUN pip install -e yuzulib -e libultimate -e ultimate-gym -e ultimate-drl

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD /entrypoint.sh