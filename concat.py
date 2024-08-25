import requests

def llama3(prompt):
    """
    Sends a prompt to the Llama3.1 model and returns the response.

    Args:
        prompt (str): The prompt to send to the Llama3.1 model.

    Returns:
        str: The response from the Llama3.1 model.
    """
    url = "http://localhost:11434/api/chat"
    data = {
        "model": "llama3.1",
        "messages": [
            {
              "role": "user",
              "content": prompt
            }
        ],
        "stream": False
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, headers=headers, json=data)
    return(response.json()['message']['content'])