from sqlmodel import create_engine, SQLModel

def create_db_engine(db_path: str):
    """
    Creates and returns a SQLite engine.
    Also creates tables if they don't exist.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    return engine
