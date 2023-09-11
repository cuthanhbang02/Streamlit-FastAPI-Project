uvicorn app.main:app --host localhost --port 8000 --reload

Create main.py

pip install fastapi[all]

Setting up Environment Variables .env -> config.py

Connect to db -> database.py

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

Installing the UUID OSSP PostgreSQL Extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

Create Database Models with SQLAlchemy in FastAPI -> model.py

Creating Schemas with Pydantic in FastAPI -> schemas.py

pip install "passlib[bcrypt]"

Hash and verify password-> utils.py

pip install 'fastapi-jwt-auth[asymmetric]'

Configure the application to use public and private keys, the RS256 algorithm, and to use the tokens in the cookies -> oauth2.py

Creating the Controllers -> router folder
authentication control: Signin, register, refresh token, logout -> auth.py

user control: user.py

calo control: calo.py

Database Migration with Alembic
pip install alembic
alembic init alembic

alembic revision --autogenerate -m "creat users table"
alembic upgrade head
