# CustomGPT
AI Chat Bot That Talks Like You!

## Steps to start using

1) Request a copy of your messages from Discord.
2) Drag the `messages` folder into the root of the project and run
    ```
    python make_data.py
    ```
3) Improvements can be made to the data by cleaning up (Work in Progress)
4) Now start training the model
    ```
    python train.py
    ```
5) Highly recommend you train on a device that has a GPU. 

    ```
    For benchmark, 
    it took about 15 minutes to train on a RTX 2060 with 102k lines of messages 
    ```
6) After its done, results.txt will contain newly generated data that "talks" like you!