"""Database connection and utilities for MongoDB."""

import logging
from typing import Optional
from urllib.parse import quote_plus

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import (
    ConnectionFailure,
    OperationFailure,
    ServerSelectionTimeoutError,
)

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class Database:
    """MongoDB database connection manager."""

    client: Optional[AsyncIOMotorClient] = None
    database: Optional[AsyncIOMotorDatabase] = None

    @classmethod
    def get_connection_string(cls) -> str:
        """Build MongoDB connection string from settings."""
        username = quote_plus(settings.mongodb_username)
        password = quote_plus(settings.mongodb_password)
        cluster = settings.mongodb_cluster
        app_name = settings.mongodb_app_name

        return (
            f"mongodb+srv://{username}:{password}@{cluster}/"
            f"?retryWrites=true&w=majority&appName={app_name}"
        )

    @classmethod
    async def connect(cls):
        """Connect to MongoDB."""
        try:
            # Validate password is set
            if not settings.mongodb_password:
                error_msg = (
                    "MongoDB password is empty! Please set MONGODB_PASSWORD or MONGO_PASSWORD "
                    "in your .env file or environment variables."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Validate username
            if not settings.mongodb_username:
                error_msg = "MongoDB username is empty!"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Log connection attempt (without exposing password)
            logger.info("Attempting to connect to MongoDB:")
            logger.info(f"  Cluster: {settings.mongodb_cluster}")
            logger.info(f"  Username: {settings.mongodb_username}")
            logger.info(f"  Database: {settings.mongodb_database}")
            logger.info(
                f"  Password length: {len(settings.mongodb_password)} characters"
            )

            connection_string = cls.get_connection_string()

            # Log connection string format (masked password)
            masked_uri = (
                connection_string.split("@")[0].split(":")[0]
                + ":***@"
                + connection_string.split("@")[1]
            )
            logger.debug(f"Connection string: {masked_uri}")

            cls.client = AsyncIOMotorClient(
                connection_string,
                serverSelectionTimeoutMS=10000,  # Increased timeout
            )
            # Test connection
            await cls.client.admin.command("ping")
            database_name: str = settings.mongodb_database
            cls.database = cls.client[database_name]
            logger.info(f"Successfully connected to MongoDB! Database: {database_name}")
        except OperationFailure as e:
            error_code = getattr(e, "code", None)
            error_msg = str(e)
            logger.error("=" * 60)
            logger.error("MONGODB AUTHENTICATION FAILED")
            logger.error("=" * 60)
            logger.error(f"Error: {error_msg}")
            logger.error(f"Error code: {error_code}")
            logger.error("")
            logger.error("TROUBLESHOOTING STEPS:")
            logger.error("1. Verify your MongoDB Atlas username:")
            logger.error(f"   Current username: '{settings.mongodb_username}'")
            logger.error("   → Check MongoDB Atlas → Database Access → Database Users")
            logger.error("   → The username might be different from what you expect")
            logger.error("")
            logger.error("2. Verify your MongoDB Atlas password:")
            logger.error(
                "   → Make sure MONGODB_PASSWORD or MONGO_PASSWORD is set in .env"
            )
            logger.error(
                "   → If password contains special characters, ensure they're properly encoded"
            )
            logger.error("   → Try resetting the password in MongoDB Atlas if unsure")
            logger.error("")
            logger.error("3. Check database user exists:")
            logger.error("   → MongoDB Atlas → Database Access")
            logger.error("   → Ensure user exists and is active")
            logger.error("")
            logger.error("4. Verify connection string format:")
            logger.error(
                "   → Format should be: mongodb+srv://USERNAME:PASSWORD@CLUSTER/"
            )
            logger.error(
                "   → Password should be URL-encoded if it contains special chars"
            )
            logger.error("")
            logger.error("5. Test with MongoDB Atlas connection string:")
            logger.error("   → MongoDB Atlas → Connect → Drivers")
            logger.error(
                "   → Copy the connection string and compare with your settings"
            )
            logger.error("=" * 60)
            raise
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.error(
                "Please check:\n"
                "  1. MongoDB password is set correctly in .env file\n"
                "  2. Your IP address is whitelisted in MongoDB Atlas\n"
                "  3. Network connectivity to MongoDB Atlas"
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            logger.exception("Full error traceback:")
            raise

    @classmethod
    async def disconnect(cls):
        """Disconnect from MongoDB."""
        if cls.client:
            cls.client.close()
            logger.info("Disconnected from MongoDB")

    @classmethod
    def get_collection(cls, collection_name: str):
        """Get a MongoDB collection."""
        if cls.database is None:
            raise RuntimeError("Database not connected. Call Database.connect() first.")
        return cls.database[collection_name]  # type: ignore[index]


# Collection names
SUBJECTS_COLLECTION = "subjects"
STUDENTS_COLLECTION = "students"
SUBJECT_INSTANCES_COLLECTION = "subject_instances"
