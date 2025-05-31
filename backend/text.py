import requests
import json
import os
import re

def score_presentation_engagement(transcript, model="llama3"):
    """
    Uses Ollama to score a presentation transcription based on engagement and cohesiveness.
    
    Args:
        transcript (str): The transcription of the presentation so far
        model (str): The Ollama model to use (default: "llama3")
        
    Returns:
        int: A score from 1 to 100 representing how engaging and cohesive the presentation is
    """
    # Check if transcript is empty or too short
    if not transcript or len(transcript.strip()) < 10:
        return 0
        
    # Prepare the prompt for Ollama
    system_prompt = """You are a simulated audience of a presentation. Your task is to "react" and "rate" a presentation in terms of your current engagement, given the transcription (what was said) during the presentation so far. Keep in mind that the transcription that is given may be inaccurate to what was actually said because of the microphone. 
    Score it on engagement and cohesiveness from 1 to 100, where:
    - 1-20: Very poor, disjointed, not engaging at all
    - 21-40: Below average, lacks structure and engagement
    - 41-60: Average, somewhat cohesive and moderately engaging
    - 61-80: Good, mostly cohesive with engaging elements
    - 81-100: Excellent, highly cohesive and extremely engaging
    
    IMPORTANT: Return ONLY an integer score from 1 to 100. Do not include any explanation, 
    comments, or additional text.
    """
    
    user_prompt = f"Here's a presentation transcript. Score it from 1 to 100 based on engagement and cohesiveness:\n\n{transcript}"
    
    # Prepare the request to Ollama API
    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    
    try:
        # Make the API request to Ollama
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the response
        result = response.json()
        response_text = result["message"]["content"].strip()
        
        # Extract just the integer score from the response
        # This handles cases where the model might add some explanation despite instructions
        match = re.search(r'\b([1-9][0-9]?|100)\b', response_text)
        if match:
            score = int(match.group(1))
            return score
            
        # If no valid score found but there's a number in the response, try to use that
        if response_text.isdigit() and 1 <= int(response_text) <= 100:
            return int(response_text)
            
        # Default score if parsing fails
        print(f"Failed to parse score from Ollama response: {response_text}")
        return 50
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama. Make sure Ollama is running on localhost:11434")
        return 0
    except Exception as e:
        print(f"Error scoring presentation: {str(e)}")
        return 0

def get_presentation_feedback(transcript):
    """
    Get more detailed feedback on a presentation transcript using Ollama.
    
    Args:
        transcript (str): The presentation transcript
        
    Returns:
        dict: Feedback with scores and suggestions
    """
    # Get the engagement score
    engagement_score = score_presentation_engagement(transcript)
    
    # For now, just return the engagement score
    # This function can be expanded to provide more detailed feedback
    return {
        "engagement_score": engagement_score,
        "suggestions": []  # Placeholder for future detailed feedback
    }