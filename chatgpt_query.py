import requests
import os

def query_chatgpt(prompt):
    api_key = os.getenv('CHATGPT_API_KEY')
    if not api_key:
        raise ValueError("API key not found. Please set the CHATGPT_API_KEY environment variable.")
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",  # Update the model ID as required
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=data)
    #print(response.text)
    return response.json()

def main():
    prompt = "Tell me what is the abstraction and reasoning corpus"
    
    try:
        result = query_chatgpt(prompt)
        if result.get('choices'):
            print(result['choices'][0]['message']['content'])
        else:
            print("No response received.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
