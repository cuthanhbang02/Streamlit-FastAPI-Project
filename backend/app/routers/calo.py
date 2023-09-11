from datetime import datetime
import uuid
from .. import schemas, models
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status, APIRouter, Response
from ..database import get_db
from app.oauth2 import require_user

router = APIRouter()

# Get calories
@router.get('/', response_model=schemas.ListCaloResponse)
def get_calories(db: Session = Depends(get_db), limit: int = 10, page: int = 1, search: str = ''):
    skip = (page - 1) * limit

    calories = db.query(models.Calo).group_by(models.Calo.id).filter(
        models.Calo.created_date.contains(search)).all()
    print(calories)
    return {'calories': calories}

# Create calories
@router.post('/', status_code=status.HTTP_201_CREATED, response_model=schemas.CaloResponse)
def create_calo(calo: schemas.CreateCaloSchema, db: Session = Depends(get_db)):
    new_calo = models.Calo(**calo.dict())
    calo_query = db.query(models.Calo).filter(models.Calo.created_date == new_calo.created_date)
    exist_calo = calo_query.first()
    if exist_calo:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f'A record with this date: {new_calo.created_date} already found')
    new_calo.calo_diff = new_calo.calo_in - new_calo.calo_out
    db.add(new_calo)
    db.commit()
    db.refresh(new_calo)
    return new_calo

# Update calories
@router.put('/{date}', response_model=schemas.CaloResponse)
def update_calo(date: str, calo: schemas.UpdateCaloSchema, db: Session = Depends(get_db)):
    date = datetime.strptime(date, "%Y-%m-%d")
    date = datetime.strftime(date, "%m/%d/%Y")
    calo_query = db.query(models.Calo).filter(models.Calo.created_date == date)
    updated_calo = calo_query.first()
    if not updated_calo:
        raise HTTPException(status_code=status.HTTP_200_OK,
                            detail=f'No calories record with this date: {date} found')
    calo_query.update(calo.dict(exclude_unset=True), synchronize_session=False)
    db.commit()
    return updated_calo

# Delete calories
@router.delete('/{date}')
def delete_calo(date: str, db: Session = Depends(get_db)):
    date = datetime.strptime(date, "%Y-%m-%d")
    date = datetime.strftime(date, "%m/%d/%Y")
    calo_query = db.query(models.Calo).filter(models.Calo.created_date == date)
    calo = calo_query.first()
    if not calo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'No calories record with this date: {date} found')
    calo_query.delete(synchronize_session=False)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)