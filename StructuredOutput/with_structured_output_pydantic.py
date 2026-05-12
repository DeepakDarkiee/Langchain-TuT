from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

class Review(BaseModel):
    food: str = Field(description="The food quality")
    service: str = Field(description="The service quality")
    rating: int = Field(description="The overall rating")
    sentiment: Literal["positive", "neutral", "negative"] = Field(description="The sentiment of the review")
    summary: str = Field(description="A brief summary of the review")
    key_themes: list[str] = Field(description="The key themes mentioned in the review")


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Food Garage offers an amazing dining experience with delicious food and quick service. 
The burgers were juicy, the fries were perfectly crispy, and the shakes tasted fresh and rich. 
The staff was polite, friendly, and handled the orders professionally even during busy hours. 
The restaurant atmosphere felt energetic and modern, making it a great place to hang out with friends. 
The food quality was excellent for the price, and the order arrived hot and well-packed. 
Overall, I had a very satisfying experience and would definitely visit again.""")

print(result)