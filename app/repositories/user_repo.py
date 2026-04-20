from sqlalchemy.orm import Session
from app.models.user import User
from app.core.security import get_password_hash

class UserRepo:
    def get(self, db: Session, id: int):
        return db.query(User).filter(User.id == id).first()

    def get_by_email(self, db: Session, email: str):
        return db.query(User).filter(User.email == email).first()

    def create(self, db: Session, obj_in: dict):
        db_obj = User(
            email=obj_in["email"],
            hashed_password=get_password_hash(obj_in["password"]),
            full_name=obj_in.get("full_name"),
            is_superuser=obj_in.get("is_superuser", False)
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

user_repo = UserRepo()
