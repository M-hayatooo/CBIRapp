from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import Column
from sqlalchemy.types import Integer, String, Text

engine = create_engine("postgresql+psycopg2://postgres.fnoprhmpeqtrnmpsblfu:Qu7iurPaeaB4yWR@aws-0-ap-northeast-1.pooler.supabase.com:5432/postgres")

# user=postgres.fnoprhmpeqtrnmpsblfu password=[YOUR-PASSWORD] host=aws-0-ap-northeast-1.pooler.supabase.com port=6543 dbname=postgres

Base = declarative_base()

class BrainMRI(Base):
    __tablename__ = "brain_mris"  # テーブル名を指定
    id = Column(Integer, primary_key=True)
    featuer_rep = Column(Text())
    img_path = Column(Text())

def get_all_brain_mri():
    SessionClass = sessionmaker(engine)  # セッションを作るクラスを作成
    session = SessionClass()
    mris = session.query(BrainMRI).all()  # userテーブルの全レコードをクラスが入った配列で返す
    # mri = session.query(BrainMRI).first()  # userテーブルの最初のレコードをクラスで返す

    return mris
