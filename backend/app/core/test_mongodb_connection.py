"""Test script to diagnose MongoDB connection issues."""

import asyncio
import logging
import os
import sys
from urllib.parse import quote_plus

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, OperationFailure

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def test_connection():
    """Test MongoDB connection with detailed diagnostics."""
    print("=" * 70)
    print("MongoDB Connection Diagnostic Tool")
    print("=" * 70)
    print()

    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    # Get credentials
    username = os.getenv("MONGODB_USERNAME") or os.getenv("MONGO_USERNAME") or "230595"
    password = os.getenv("MONGODB_PASSWORD") or os.getenv("MONGO_PASSWORD") or ""
    cluster = (
        os.getenv("MONGODB_CLUSTER")
        or os.getenv("MONGO_CLUSTER")
        or "schcalc.thdwjpq.mongodb.net"
    )
    app_name = os.getenv("MONGODB_APP_NAME") or os.getenv("MONGO_APP_NAME") or "SchCalc"

    print("Configuration:")
    print(f"  Username: {username}")
    print(f"  Password: {'*' * len(password) if password else '(EMPTY!)'}")
    print(f"  Cluster: {cluster}")
    print(f"  App Name: {app_name}")
    print()

    if not password:
        print("ERROR: Password is empty!")
        print("Please set MONGODB_PASSWORD or MONGO_PASSWORD in your .env file")
        return False

    # Build connection string
    encoded_username = quote_plus(username)
    encoded_password = quote_plus(password)
    connection_string = (
        f"mongodb+srv://{encoded_username}:{encoded_password}@{cluster}/"
        f"?retryWrites=true&w=majority&appName={app_name}"
    )

    # Show masked connection string
    masked_uri = f"mongodb+srv://{username}:***@{cluster}/?retryWrites=true&w=majority&appName={app_name}"
    print(f"Connection string: {masked_uri}")
    print()

    print("Attempting connection...")
    try:
        client: AsyncIOMotorClient = AsyncIOMotorClient(
            connection_string,
            serverSelectionTimeoutMS=10000,
        )

        # Test connection
        result = await client.admin.command("ping")
        print("✓ SUCCESS! Connected to MongoDB!")
        print(f"  Ping result: {result}")

        # List databases
        db_list = await client.list_database_names()
        print(f"  Available databases: {db_list}")

        client.close()
        return True

    except OperationFailure as e:
        print()
        print("✗ AUTHENTICATION FAILED")
        print(f"  Error: {e}")
        print()
        print("Common causes:")
        print("  1. Wrong username - Check MongoDB Atlas → Database Access")
        print("  2. Wrong password - Verify password in .env matches Atlas")
        print("  3. User doesn't exist - Create user in MongoDB Atlas")
        print("  4. Password has special characters - Ensure proper URL encoding")
        print()
        print("To fix:")
        print("  1. Go to MongoDB Atlas → Database Access")
        print("  2. Check the exact username (might be different from '230595')")
        print("  3. Click 'Edit' on the user → 'Edit Password'")
        print("  4. Set a new password and update your .env file")
        return False

    except ConnectionFailure as e:
        print()
        print("✗ CONNECTION FAILED")
        print(f"  Error: {e}")
        print()
        print("Common causes:")
        print("  1. IP address not whitelisted - MongoDB Atlas → Network Access")
        print("  2. Network/firewall blocking connection")
        print("  3. Wrong cluster address")
        return False

    except Exception as e:
        print()
        print(f"✗ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)
