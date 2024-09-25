from pydantic import BaseModel, Field, model_validator
from typing import Union, ClassVar


# 定义不同的贷款类型类，继承自 BaseModel
class HomeLoan(BaseModel):
    interest_rate: float = 0.09

class PersonalLoan(BaseModel):
    interest_rate: float = 0.10

class CommercialLoan(BaseModel):
    interest_rate: float = 0.11


# 定义 Loan 模型
class Loan(BaseModel):
    loan_amount: float = Field(..., gt=0, description="贷款金额，必须大于0")
    loan_type: Union[HomeLoan, PersonalLoan, CommercialLoan]
    loan_type_mapping: ClassVar[dict[str, type]] = {
        'home': HomeLoan,
        'personal': PersonalLoan,
        'commercial': CommercialLoan
    }

    @model_validator(mode='before')
    def validate_loan_type(cls, values):
        loan_type = values.get('loan_type')
        if isinstance(loan_type, str):
            # 通过类名访问 loan_type_mapping
            loan_class = cls.loan_type_mapping.get(loan_type.lower(), None)
            if loan_class is None:
                raise ValueError(f"Invalid loan type: {loan_type}")
            values['loan_type'] = loan_class()  # 创建类实例
        elif not isinstance(loan_type, (HomeLoan, PersonalLoan, CommercialLoan)):
            raise ValueError(f"Invalid loan type instance: {loan_type}")
        return values

    # 计算利息的方法
    def calculate_interest(self):
        return self.loan_amount * self.loan_type.interest_rate


# 示例用法
loan1 = Loan(loan_amount=100000, loan_type="home")  # 字符串类型
print(f"Home Loan interest: {loan1.calculate_interest():.2f}")  # 输出：9000.00

loan2 = Loan(loan_amount=50000, loan_type=PersonalLoan())  # 直接传入类实例
print(f"Personal Loan interest: {loan2.calculate_interest():.2f}")  # 输出：5000.00

# 测试不合法的数据类型
try:
    loan3 = Loan(loan_amount=75000, loan_type=123)  # 整型，不合法
except ValueError as e:
    print(f"Error: {e}")  # 输出错误信息
