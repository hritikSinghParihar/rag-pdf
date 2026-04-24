import sys
import logging
from app.models import SessionLocal
from app.repositories.user_repo import user_repo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_admin(email, password, full_name):
    db = SessionLocal()
    try:
        user = user_repo.get_by_email(db, email=email)
        if user:
            logger.info(f"User {email} already exists.")
            return
        
        user_in = {
            "email": email,
            "password": password,
            "full_name": full_name,
            "is_superuser": True
        }
        user_repo.create(db, obj_in=user_in)
        logger.info(f"Successfully created admin user: {email}")
    except Exception as e:
        logger.error(f"Error creating admin user: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/create_admin.py <email> <password> [<full_name>]")
        sys.exit(1)
    
    email = sys.argv[1]
    password = sys.argv[2]
    full_name = sys.argv[3] if len(sys.argv) > 3 else "Admin User"
    
    create_admin(email, password, full_name)
