import os
import re
from pydantic import BaseModel, Field
from typing import Optional

# Try to import ollama package, fall back to requests if not available
try:
    import ollama
    OLLAMA_PACKAGE_AVAILABLE = True
except ImportError:
    import requests
    import json
    OLLAMA_PACKAGE_AVAILABLE = False

# Define Pydantic model for structured output
class PresentationScore(BaseModel):
    score: int = Field(..., ge=1, le=100, description="Engagement and cohesiveness score from 1-100")


def score_presentation_engagement(transcript, model="llama3:8b"):
    """
    Uses Ollama to score a presentation transcription based on engagement and cohesiveness.
    
    Args:
        transcript (str): The transcription of the presentation so far
        model (str): The Ollama model to use (default: "llama3:8b")
        
    Returns:
        int: A score from 1 to 100 representing how engaging and cohesive the presentation is
    """
    # Check if transcript is empty or too short
    if not transcript or len(transcript.strip()) < 10:
        return 0
        
    # Prepare the system prompt
    system_prompt = """You are a simulated audience of a presentation. Your task is to "react" and "rate" a presentation in terms of your current engagement, given the transcription (what was said) during the presentation so far. Keep in mind that the transcription that is given may be inaccurate to what was actually said because of the microphone. 
    Score it on engagement and cohesiveness from 1 to 100, where:
    - 1-20: Very poor, disjointed, not engaging at all
    - 21-40: Below average, lacks structure and engagement
    - 41-60: Average, somewhat cohesive and moderately engaging
    - 61-80: Good, mostly cohesive with engaging elements
    - 81-100: Excellent, highly cohesive and extremely engaging
    """
    
    user_prompt = f"Here's a presentation transcript. Score it from 1 to 100 based on engagement and cohesiveness:\n\n{transcript}"
    
    try:
        # Use the ollama package if available, which supports structured output
        if OLLAMA_PACKAGE_AVAILABLE:
            try:
                # Use the structured output feature of the ollama package
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    format=PresentationScore.model_json_schema(),
                )
                
                # Parse the structured response
                score_data = PresentationScore.model_validate_json(response.message.content)
                return score_data.score
                
            except Exception as e:
                print(f"Error using ollama package structured output: {str(e)}")
                # Fall back to requests method if the structured output fails
        
        # Fallback to using requests directly
        url = "http://localhost:11434/api/chat"
        
        # Add schema to system prompt for fallback method
        format_instructions = """
        IMPORTANT: Your response MUST be a valid JSON object with this EXACT structure:
        {"score": INTEGER}
        
        Where INTEGER is a number from 1 to 100. Do not include any explanation,
        comments, or additional text outside of this JSON structure.
        """
        
        fallback_system_prompt = system_prompt + "\n" + format_instructions
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": fallback_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
        
        # Make the API request to Ollama
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the response
        result = response.json()
        response_text = result["message"]["content"].strip()
        
        # Try to parse as JSON first
        try:
            # Extract JSON if embedded in other text
            json_match = re.search(r'\{[^\{\}]*"score"\s*:\s*\d+[^\{\}]*\}', response_text)
            if json_match:
                response_text = json_match.group(0)
                
            # Parse the JSON
            response_json = json.loads(response_text)
            if 'score' in response_json and isinstance(response_json['score'], int):
                score = response_json['score']
                if 1 <= score <= 100:
                    print(f"text score: {score}")
                    return score
        except json.JSONDecodeError:
            pass
            
        # Fallback to regex extraction
        match = re.search(r'\b([1-9][0-9]?|100)\b', response_text)
        if match:
            score = int(match.group(1))
            return score
            
        # Last resort: check if the entire response is a digit
        if response_text.isdigit() and 1 <= int(response_text) <= 100:
            return int(response_text)
            
        # Default score if all parsing methods fail
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


if __name__ == "__main__":
    score = score_presentation_engagement("Hello everyone. Thank you for joining me today. I'm excited to share with you some insights about the emerging role of artificial intelligence in modern healthcare.")
    print(score)