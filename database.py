from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import Column
from sqlalchemy.types import Integer, String, Text

engine = create_engine("postgresql+psycopg2://postgres.fnoprhmpeqtrnmpsblfu:Qu7iurPaeaB4yWR@aws-0-ap-northeast-1.pooler.supabase.com:5432/postgres")
Base = declarative_base()


class BrainMRI(Base):
    __tablename__ = "brain_mris"  # テーブル名を指定
    id = Column(Integer, primary_key=True)
    feature_rep = Column(Text())  # featuer_rep = Column(Text())
    img_path = Column(Text())
    clinical_info_path = Column(Text())
    uid = Column(Integer())


def get_all_brain_mri():
    SessionClass = sessionmaker(engine)  # セッションを作るクラスを作成
    session = SessionClass()
    mris = session.query(BrainMRI).all()  # userテーブルの全レコードをクラスが入った配列で返す
    # mri = session.query(BrainMRI).first()  # userテーブルの最初のレコードをクラスで返す
    result = session.query(BrainMRI).filter(BrainMRI.uid == 1).first()
    
    low_dimentional_representations = session.query(BrainMRI.uid).all()
    # 取得したuidのリストを表示
    for ldr in low_dimentional_representations:
        print(ldr[0])

    if result:
        latent_value = result.latent
        print("Latent value:", latent_value)
    else:
        print("No record found with uid 1.")

    return mris

def get_all_ldr():
    SessionClass = sessionmaker(engine)  # セッションを作るクラスを作成
    session = SessionClass()    
    ldrs= session.query(BrainMRI.clinical_info_path, BrainMRI.feature_rep).all()
    # ldrs= session.query(BrainMRI.id, BrainMRI.feature_rep).all() スペルミス
    return ldrs

# def get_ie_cbir_model_weight():
#     SessionClass = sessionmaker(engine)  # セッションを作るクラスを作成
#     session = SessionClass()
#     with open(session, 'wb+') as f:
#         res = supabase.storage.from_('bucket_name').download(source)
#         f.write(res)
#     return res
