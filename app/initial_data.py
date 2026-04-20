import logging
from app.models import SessionLocal, Base, engine
from app.repositories.user_repo import user_repo
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    db = SessionLocal()
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        
        # Check if superuser exists
        user = user_repo.get_by_email(db, email=settings.FIRST_SUPERUSER)
        if not user:
            logger.info(f"Creating superuser {settings.FIRST_SUPERUSER}")
            user_in = {
                "email": settings.FIRST_SUPERUSER,
                "password": settings.FIRST_SUPERUSER_PASSWORD,
                "full_name": "Initial Admin",
                "is_superuser": True
            }
            user_repo.create(db, obj_in=user_in)
        else:
            logger.info(f"Superuser {settings.FIRST_SUPERUSER} already exists")
    finally:
        db.close()

if __name__ == "__main__":
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized.")
