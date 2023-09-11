from datetime import datetime
from typing import List
import uuid
from pydantic import BaseModel, EmailStr, constr


class UserBaseSchema(BaseModel):
    name: str
    email: EmailStr

    class Config:
        orm_mode = True


class CreateUserSchema(UserBaseSchema):
    password: constr(min_length=8)
    passwordConfirm: str
    role: str = 'user'
    verified: bool = False


class LoginUserSchema(BaseModel):
    email: EmailStr
    password: constr(min_length=8)


class UserResponse(UserBaseSchema):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    

class CaloBaseSchema(BaseModel):
    created_date: str
    calo_in: float
    calo_out: float
    calo_diff: float

    class Config:
        orm_mode = True


class CreateCaloSchema(CaloBaseSchema):
    pass


class CaloResponse(CaloBaseSchema):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class UpdateCaloSchema(BaseModel):
    calo_in: float
    calo_out: float
    calo_diff: float
    created_at: datetime | None = None
    updated_at: datetime | None = None

    class Config:
        orm_mode = True


class ListCaloResponse(BaseModel):
    calories: List[CaloResponse] | None = None


