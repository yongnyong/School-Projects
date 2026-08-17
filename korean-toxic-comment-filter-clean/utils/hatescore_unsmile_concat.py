import pandas as pd

# 1) HateScore 로드
hatescore = pd.read_csv("./raw_data/HateScore.csv")  # 인코딩 문제 있으면 encoding="utf-8-sig" 써도 됨

print(hatescore.head())
print(hatescore['macrolabel'].value_counts())
print(hatescore['microlabel'].value_counts())

# 최종 사용할 레이블 컬럼
label_cols = [
    "여성/가족", "남성", "성소수자", "인종/국적",
    "연령", "지역", "종교", "기타혐오",
    "악플/욕설", "clean",
]

# 새 데이터프레임 뼈대
hs_proc = pd.DataFrame()
hs_proc["문장"] = hatescore["comment"]
for c in label_cols:
    hs_proc[c] = 0


def fill_labels(row):
    macro = row["macrolabel"]
    micro = row["microlabel"]

    labels = {
        "여성/가족": 0,
        "남성": 0,
        "성소수자": 0,
        "인종/국적": 0,
        "연령": 0,
        "지역": 0,
        "종교": 0,
        "기타혐오": 0,
        "악플/욕설": 0,
        "clean": 0,
    }

    # 1) 혐오발언인 경우: microlabel에 따라 타겟 선택
    if macro == "혐오발언":
        if micro in ["여성", "여성/가족"]:
            labels["여성/가족"] = 1
        elif micro == "남성":
            labels["남성"] = 1
        elif micro == "성소수자":
            labels["성소수자"] = 1
        elif micro == "인종/국적":
            labels["인종/국적"] = 1
        elif micro == "연령":
            labels["연령"] = 1
        elif micro == "지역":
            labels["지역"] = 1
        elif micro == "종교":
            labels["종교"] = 1
        elif micro in ["기타혐오", "기타", "기타 혐오"]:
            labels["기타혐오"] = 1

    # 2) 단순 악플
    elif macro in ["단순악플", "단순 악플"]:
        labels["악플/욕설"] = 1

    # 3) 일반/중립/클린 계열
    elif macro in ["중립", "일반 댓글", "일반", "clean", "None"]:
        labels["clean"] = 1

    # 필요하면 기타 macro 값도 처리 (예: 누락 케이스)
    return pd.Series(labels)

hs_labels = hatescore.apply(fill_labels, axis=1)
for c in label_cols:
    hs_proc[c] = hs_labels[c]

hs_proc["source"] = "hatescore"

# UnSmile 로드
unsmile_train = pd.read_csv("./raw_data/unsmile_train_v1.0.tsv", sep="\t")
unsmile_valid = pd.read_csv("./raw_data/unsmile_valid_v1.0.tsv", sep="\t")

unsmile = pd.concat([unsmile_train, unsmile_valid], ignore_index=True)

# 컬럼 이름 통일 (기타 혐오 -> 기타혐오)
unsmile = unsmile.rename(columns={"기타 혐오": "기타혐오"})

# 최종 사용할 컬럼만 추출
unsmile_proc = unsmile[["문장"] + label_cols].copy()
unsmile_proc["source"] = "unsmile"

combined = pd.concat([unsmile_proc, hs_proc], ignore_index=True)

print(combined.head())
print(combined["source"].value_counts())

combined.to_csv("combined_unsmile_hatescore.csv", index=False, encoding="utf-8-sig")