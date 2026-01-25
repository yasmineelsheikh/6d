from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from sqlalchemy import create_engine, Column, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import bcrypt
import os
from cryptography.fernet import Fernet
import base64

# Database setup
# Use Supabase PostgreSQL if available, otherwise fall back to SQLite for local dev
POSTGRES_URL = os.getenv("POSTGRES_URL_NON_POOLING") or os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")

# Check if we're on Vercel (should always use PostgreSQL there)
is_vercel = os.getenv("VERCEL") == "1"

if POSTGRES_URL and (POSTGRES_URL.startswith("postgres") or is_vercel):
    # Use Supabase PostgreSQL
    if not POSTGRES_URL.startswith("postgres"):
        # If on Vercel but POSTGRES_URL not found, raise an error
        raise ValueError(
            "PostgreSQL connection string required on Vercel. "
            "Please set POSTGRES_URL_NON_POOLING or POSTGRES_URL environment variable."
        )
    # Normalize connection string: SQLAlchemy prefers postgresql:// over postgres://
    DATABASE_URL = POSTGRES_URL.replace("postgres://", "postgresql://", 1)
    # PostgreSQL doesn't need check_same_thread
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
else:
    # Fallback to SQLite for local development only
    db_path = "./users.db"
    DATABASE_URL = f"sqlite:///{db_path}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User model
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    hashed_password = Column(String)
    credits = Column(String, default="1000")  # Store as string to handle large numbers
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to credentials
    credentials = relationship("UserCredentials", back_populates="user", cascade="all, delete-orphan")

# UserCredentials model for storing S3 credentials
class UserCredentials(Base):
    __tablename__ = "user_credentials"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    credential_type = Column(String, nullable=False)  # "s3", "huggingface", etc.
    encrypted_access_key = Column(Text)  # Encrypted AWS access key
    encrypted_secret_key = Column(Text)  # Encrypted AWS secret key
    bucket_name = Column(String)  # S3 bucket name
    region = Column(String, default="us-east-1")  # AWS region
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to user
    user = relationship("User", back_populates="credentials")

# Create tables if they don't exist
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Warning: Could not create database tables at startup: {e}")

# JWT settings - use Supabase JWT secret if available, otherwise fall back to custom secret
SECRET_KEY = os.getenv("SUPABASE_JWT_SECRET") or os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )

def get_password_hash(password: str) -> str:
    """Hash a password."""
    # Check password length (bcrypt limit is 72 bytes)
    password_bytes = len(password.encode('utf-8'))
    if password_bytes > 72:
        raise ValueError("Password cannot be longer than 72 bytes")
    
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str):
    """Decode and verify a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def get_user_by_email(db: Session, email: str):
    """Get a user by email."""
    return db.query(User).filter(User.email == email).first()

def get_user_by_id(db: Session, user_id: str):
    """Get a user by ID."""
    return db.query(User).filter(User.id == user_id).first()

def create_user(db: Session, email: str, first_name: str, last_name: str, password: str):
    """Create a new user with 1000 free trial credits."""
    import uuid
    hashed_password = get_password_hash(password)
    user = User(
        id=str(uuid.uuid4()),
        email=email,
        first_name=first_name,
        last_name=last_name,
        hashed_password=hashed_password,
        credits="1000"  # Free trial credits
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

# Encryption/Decryption helpers for credentials
def _get_encryption_key() -> bytes:
    """Get encryption key from SECRET_KEY."""
    # Use SECRET_KEY to derive a Fernet key
    key = SECRET_KEY.encode('utf-8')
    # Pad or truncate to 32 bytes for Fernet
    if len(key) < 32:
        key = key.ljust(32, b'0')
    elif len(key) > 32:
        key = key[:32]
    # Fernet requires base64-encoded 32-byte key
    return base64.urlsafe_b64encode(key)

def encrypt_credential(plaintext: str) -> str:
    """Encrypt a credential value."""
    if not plaintext:
        return ""
    f = Fernet(_get_encryption_key())
    encrypted = f.encrypt(plaintext.encode('utf-8'))
    return encrypted.decode('utf-8')

def decrypt_credential(encrypted: str) -> str:
    """Decrypt a credential value."""
    if not encrypted:
        return ""
    try:
        f = Fernet(_get_encryption_key())
        decrypted = f.decrypt(encrypted.encode('utf-8'))
        return decrypted.decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to decrypt credential: {str(e)}")

def get_user_s3_credentials(db: Session, user_id: str) -> Optional[dict]:
    """Get user's S3 credentials (decrypted)."""
    cred = db.query(UserCredentials).filter(
        UserCredentials.user_id == user_id,
        UserCredentials.credential_type == "s3"
    ).first()
    if not cred:
        return None
    return {
        "access_key": decrypt_credential(cred.encrypted_access_key) if cred.encrypted_access_key else None,
        "secret_key": decrypt_credential(cred.encrypted_secret_key) if cred.encrypted_secret_key else None,
        "bucket_name": cred.bucket_name,
        "region": cred.region
    }

def save_user_s3_credentials(
    db: Session,
    user_id: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    region: str = "us-east-1"
) -> UserCredentials:
    """Save or update user's S3 credentials (encrypted)."""
    import uuid
    cred = db.query(UserCredentials).filter(
        UserCredentials.user_id == user_id,
        UserCredentials.credential_type == "s3"
    ).first()
    
    if cred:
        # Update existing
        cred.encrypted_access_key = encrypt_credential(access_key)
        cred.encrypted_secret_key = encrypt_credential(secret_key)
        cred.bucket_name = bucket_name
        cred.region = region
        cred.updated_at = datetime.utcnow()
    else:
        # Create new
        cred = UserCredentials(
            id=str(uuid.uuid4()),
            user_id=user_id,
            credential_type="s3",
            encrypted_access_key=encrypt_credential(access_key),
            encrypted_secret_key=encrypt_credential(secret_key),
            bucket_name=bucket_name,
            region=region
        )
        db.add(cred)
    
    db.commit()
    db.refresh(cred)
    return cred
