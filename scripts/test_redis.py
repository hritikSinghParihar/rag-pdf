import redis
from app.core.config import settings

def test_redis():
    try:
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            socket_timeout=5
        )
        ping = r.ping()
        print(f"Successfully connected to Redis. Ping: {ping}")
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")

if __name__ == "__main__":
    test_redis()
