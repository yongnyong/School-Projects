import pandas as pd
from sklearn.model_selection import train_test_split

# 1. CSV 불러오기
df = pd.read_csv("C:/Users/user/PycharmProjects/PythonProject9/deeplearning_1/data/final_converted_input_target.csv", encoding='utf-8')


# 2. train (80%) + temp (20%) 분할
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. temp → validation (10%) + test (10%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 4. 저장
train_df.to_csv('C:/Users/user/PycharmProjects/pythonProject9/deeplearning_1/data/train.csv', index=False)
val_df.to_csv('C:/Users/user/PycharmProjects/pythonProject9/deeplearning_1/data/val.csv', index=False)
test_df.to_csv('C:/Users/user/PycharmProjects/pythonProject9/deeplearning_1/data/test.csv', index=False)

print("✅ 데이터 분할 완료!")
print(f"Train: {len(train_df)}개, Val: {len(val_df)}개, Test: {len(test_df)}개")