from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

#schema for the structured output
Review = {
  "title": "Review",
  "description": "A structured representation of a restaurant review.",
  "type": "object",
  "properties": {
    "food": {
      "type": "string",
      "description": "The food quality"
    },
    "service": {
      "type": "string",
      "description": "The service quality"
    },  
    "rating": {
      "type": "integer",
      "description": "The overall rating"
    },
    "sentiment": {
      "type": "string",
      "description": "The sentiment of the review",
      "enum": ["positive", "neutral", "negative"]
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "key_themes": {
      "type": "array",
      "description": "The key themes mentioned in the review",
      "items": {
        "type": "string"
      }
    }
  },
  "required": ["food", "service", "rating", "sentiment", "summary", "key_themes"]
}


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Food Garage offers an amazing dining experience with delicious food and quick service. 
The burgers were juicy, the fries were perfectly crispy, and the shakes tasted fresh and rich. 
The staff was polite, friendly, and handled the orders professionally even during busy hours. 
The restaurant atmosphere felt energetic and modern, making it a great place to hang out with friends. 
The food quality was excellent for the price, and the order arrived hot and well-packed. 
Overall, I had a very satisfying experience and would definitely visit again.""")

print(result)