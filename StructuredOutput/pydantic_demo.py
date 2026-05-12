from pydantic import BaseModel, EmailStr, Field
from typing import Optional
class User(BaseModel):
    id: int
    name: str = "Deepak"
    email: Optional[EmailStr] = None
    age : Optional[int] = None
    cgpa: float = Field(gt=0, lt=10, description="CGPA must be between 0 and 10")


new_user = User(id=1, name="Alice", email="alice@example.com", age=98, cgpa=9.5)

print(new_user)