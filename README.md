# automat-llm
Mobile AI Assistant

## How To Run
To run this demo first ust `pip install -r requirements.txt` to install basic packages from pip. Then you will have to optionally run `pip install ./dia` for voice interaction if desired.
the use of dia is entirely optional and requirements should demonstrate the basic demo. 
Before running the demo be sure to run `huggingface-cli login` and login as needed, then run
`python main.py`, it should work. If any Weaviate issues come up contact Sasori Zero Labs (mileslitteral@sasorizerolabs.com)

## Known Issues
Some users report issues installing Langchain and Dia, this is currently being addressed, the best we can suggest for now is using venv and also being very careful to not have multiple instances of python on your system.
