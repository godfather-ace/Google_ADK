## Google's ADK Implementation

Artifacts used during Google's ADK Workshop. 

Steps: 
1. Create a project folder (uv package is recommended, else mkdir)
2. Create and activate a virtual environment 
3. Install the required libraries

```
python3 -m venv .venv
source .venv/bin/activate
pip install google-adk litellm crewai "crewai[tools]"
```
