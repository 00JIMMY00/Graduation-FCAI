### Project Tree :
   
- face_recognition


    - src

        - image_capture.py

        - model_training.py

        - facial_recognition.py

    - README.md

- voice_to_text

    - src

    - README.md

- RAG_model 

    - src

    - README.md



### Windows commands :
- create  you env

    `python -m venv --system-site-packages face_rec`

- activate you env

    `.\face_rec\Scripts\activate`

- install the required requirements 

    ` pip install -r requirements.txt`

### Linux/unix commands (same for rasspberry pi os )

- create  you env :

    - use this for raspberry pi os 

        `python3 -m venv --system-site-packages face_rec_env`

    - use .venv for linux (choose yes in the alert message on vscode)
        
        `python3 -m venv .venv`

- activate your env (if you're in vscode , no need to use this line) :

    `source face_rec_env/bin/activate`
    
    -in case the .venv isn't activated , run this 

    `source .venv/bin/activate`

- install the required livraries : 

    `pip install -r requirements.txt`