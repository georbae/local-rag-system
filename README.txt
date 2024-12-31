```markdown
# Local RAG System

## Project Overview

This project implements a local Retrieval-Augmented Generation (RAG) system, combining the power of Large Language Models (LLMs) with efficient information retrieval. The system is designed to process local documents, create embeddings, and use a local LLM to generate responses based on retrieved information.

Author: George Baez

## Features

- Multi-format document processing (PDF, DOCX)
- OCR capabilities for image-based text in PDFs
- Local LLM integration using LlamaCpp
- Vector storage with Chroma
- User-friendly interface with Streamlit

## Requirements

- Python 3.8+
- See `requirements.txt` for a full list of Python dependencies

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/local-rag-system.git
   cd local-rag-system
   ```

2. Create a virtual environment:
   ```
   python -m venv rag_env
   source rag_env/bin/activate  # On Windows use `rag_env\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up the environment variables:
   Copy the `.env.example` file to `.env` and update the values:
   ```
   cp .env.example .env
   ```
   Edit the `.env` file with your specific paths and settings.

## Development Environment Setup

### VS Code Configuration

This project includes a specific VS Code configuration to ensure a consistent development environment. 

1. Create a `.vscode` folder in your project root if it doesn't exist.

2. Inside the `.vscode` folder, create a `settings.json` file with the following content:

   ```json
   {
       "python.defaultInterpreterPath": "${workspaceFolder}/rag_env/bin/python",
       "python.terminal.activateEnvironment": true
   }

    
This configuration does two things:
	•	Sets the default Python interpreter to the one in your project's virtual environment.
	•	Ensures that VS Code automatically activates the virtual environment when you open a terminal.
Virtual Environment (rag_env)
This project uses a virtual environment named rag_env. Here's how to set it up:
	1	Create the virtual environment:     python -m venv rag_env
	2	
	3	    
Activate the virtual environment:
	•	On Windows:    rag_env\Scripts\activate
	•	
	•	    
On macOS and Linux:
    source rag_env/bin/activate

    
With the virtual environment activated, install the project dependencies:
    pip install -r requirements.txt

    
	1	VS Code should automatically detect and use this virtual environment when you open a Python file or terminal, thanks to the settings.json configuration.
Note: The rag_env folder is typically added to .gitignore to avoid committing environment-specific files to the repository. Each user should create their own virtual environment locally.
Why Use a Virtual Environment?
Using a virtual environment (rag_env) allows you to:
	•	Isolate project dependencies from your global Python installation.
	•	Ensure consistent package versions across different development machines.
	•	Easily recreate the development environment on other systems.
Remember to activate the virtual environment every time you work on the project, unless you're using VS Code which should handle this automatically with the provided settings.


## Environment Variables

The following environment variables need to be set in the `.env` file:

- `LLM_MODEL`: The name of the local LLM model file (e.g., "llama-2-7b-chat.Q8_0.gguf")
- `DIRECTORY`: The path to the folder containing documents for retrieval
- `CHROMA_DB_PATH`: The path to store the Chroma database
- `MODEL_PATH`: The path to the folder containing the LLM model file

## VS Code Settings

For optimal use with VS Code, add the following to your `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/rag_env/bin/python",
    "python.terminal.activateEnvironment": true
}
```

This ensures that VS Code uses the correct Python interpreter and activates the virtual environment automatically.

## Important Notes on Static Paths

The following sections of code may need to be updated based on your local setup:

1. Model loading:
   ```python
   model_path = os.path.join(os.getenv('MODEL_PATH', './models'), model_name)
   ```
   Ensure that the `MODEL_PATH` in your `.env` file points to the correct directory containing your LLM model.

2. Document directory:
   ```python
   directory = os.getenv('DIRECTORY')
   ```
   Update the `DIRECTORY` in your `.env` file to point to your local document folder.

3. Chroma DB path:
   ```python
   chroma_db_path = os.getenv('CHROMA_DB_PATH')
   ```
   Set the `CHROMA_DB_PATH` in your `.env` file to a suitable location for storing the vector database.

## Running the Application

To run the Streamlit app:

```
streamlit run app.py
```

## Project Structure

- `local-rag-agent`: Main application file
- `requirements.txt`: List of Python dependencies
- `.env`: Environment variables (create this from `.env.example`)
- `models/`: Directory for storing LLM models
- `README.md`: This file

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

There is currently no licensing. I only ask if you use my code for financial gain you include me on the team of developers. George Baez - georbae@amazon.com

## Acknowledgments

This project uses several open-source libraries and models. Please refer to the `requirements.txt` file and the code comments for more details on these dependencies.


```
