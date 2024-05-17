from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import Column
from sqlalchemy.types import Integer, Text

engine = create_engine("postgresql+psycopg2://postgres.fnoprhmpeqtrnmpsblfu:Qu7iurPaeaB4yWR@aws-0-ap-northeast-1.pooler.supabase.com:5432/postgres")
Base = declarative_base()


class BrainMRI(Base):
    __tablename__ = "brain_mris"  # テーブル名を指定
    id = Column(Integer, primary_key=True)
    feature_rep = Column(Text())  # featuer_rep = Column(Text())
    img_path = Column(Text())
    clinical_info_path = Column(Text())
    uid = Column(Integer())


def get_all_ldr_uid():
    SessionClass = sessionmaker(engine)  # セッションを作るクラスを作成
    session = SessionClass()    
    ldrs= session.query(BrainMRI.uid, BrainMRI.feature_rep).all()
    # ldrs= session.query(BrainMRI.id, BrainMRI.feature_rep).all() スペルミス
    return ldrs


def get_clinical_info_urls(uids):
    SessionClass = sessionmaker(engine)
    session = SessionClass()
    urls = session.query(BrainMRI.uid, BrainMRI.clinical_info_path).filter(BrainMRI.uid.in_(uids)).all()
    return urls

# def get_ie_cbir_model_weight():
#     SessionClass = sessionmaker(engine)  # セッションを作るクラスを作成
#     session = SessionClass()
#     with open(session, 'wb+') as f:
#         res = supabase.storage.from_('bucket_name').download(source)
#         f.write(res)
#     return res
