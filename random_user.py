from faker import Faker
import random
from pypinyin import pinyin, Style
from pydantic import BaseModel

# 初始化 Faker
fake = Faker('zh_CN')

# 定义用户数据模型
class RandomUser(BaseModel):
    first_name: str
    last_name: str
    email: str
    first_name_pinyin: str
    last_name_pinyin: str
    title: str

# 随机生成邮箱
def generate_random_email(domains=None):
    if domains is None:
        domains = ['163.com', '126.com', 'foxmail.com']
    domain = random.choice(domains)  # 随机选择一个邮箱后缀

    username = fake.user_name()  # 生成一个随机用户名
    random_number = random.randint(1000, 9999)  # 生成一个四位数的随机数字

    return f"{username}{random_number}@{domain}"

# 随机生成中文姓和拼音
def generate_chinese_surname_pinyin():
    surname = fake.last_name()
    pinyin_surname = ''.join([item[0] for item in pinyin(surname, style=Style.NORMAL)])
    return surname, pinyin_surname

# 随机生成中文名字和拼音
def generate_chinese_given_name_pinyin():
    given_name = fake.first_name()
    pinyin_given_name = ''.join([item[0] for item in pinyin(given_name, style=Style.NORMAL)])
    return given_name, pinyin_given_name

# 随机生成称呼
def generate_title():
    return random.choice(['Ms.', 'Mr.'])

# 生成随机用户
def generate_random_user() -> RandomUser:
    first_name, first_name_pinyin = generate_chinese_given_name_pinyin()
    last_name, last_name_pinyin = generate_chinese_surname_pinyin()
    email = generate_random_email()
    title = generate_title()

    return RandomUser(
        first_name=first_name,
        last_name=last_name,
        email=email,
        first_name_pinyin=first_name_pinyin,
        last_name_pinyin=last_name_pinyin,
        title=title
    )
