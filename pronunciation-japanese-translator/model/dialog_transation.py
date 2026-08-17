from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
import pandas as pd
import time
import json
import os

# ▶️ 경로 설정
CHROMEDRIVER_PATH = r"C:\Users\user\PycharmProjects\PythonProject9\deeplearning_1\model\chromedriver.exe"

SAVE_PATH = "translated_topic2_temp.csv"

FINAL_PATH = "translated_topic2.csv"

# ▶️ Selenium 옵션
chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--remote-debugging-port=9222")

driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=chrome_options)

# ▶️ Papago 크롤링 함수
def get_papago_output(japanese_text):
    try:
        driver.get("https://papago.naver.com/")
        input_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "textarea#txtSource"))
        )
        input_box.clear()
        input_box.send_keys(japanese_text)
        time.sleep(2)

        try:
            pron_elem = WebDriverWait(driver, 4).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#sourceEditArea p span"))
            )
            pronunciation = pron_elem.text.strip()
        except:
            pronunciation = "[발음 없음]"

        try:
            trans_elem = WebDriverWait(driver, 4).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#txtTarget span"))
            )
            translation = trans_elem.text.strip()
        except:
            translation = "[번역 없음]"

    except Exception as e:
        print(f"❌ 크롤링 실패: {e}")
        pronunciation, translation = "[에러]", "[에러]"

    return pronunciation, translation

# ▶️ topic1.json 로드 및 담화 추출
with open("topic2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

utterances = []
for d in data:
    for u in d["utterances"]:
        utterances.append({
            "topic_id": d["topic_id"],
            "dialogue_id": d["dialogue_id"],
            "speaker": u["speaker"],
            "utterance": u["utterance"]
        })

# ▶️ 중복 제거용 완료 목록 불러오기
if os.path.exists(SAVE_PATH):
    done_df = pd.read_csv(SAVE_PATH)
    done_set = set(done_df["utterance"])
    print(f"🔁 이어서 진행: {len(done_set)}개 완료됨")
else:
    done_df = pd.DataFrame()
    done_set = set()
    print("🆕 새로 시작합니다")

# ▶️ 미완료만 추출
remaining = [row for row in utterances if row["utterance"] not in done_set]

# ▶️ 진행률 표시 + 저장
with open(SAVE_PATH, "a", encoding="utf-8-sig") as f:
    for row in tqdm(remaining, desc="📘 Papago 번역 진행중", unit="문장"):
        pron, trans = get_papago_output(row["utterance"])
        result_line = f"{row['topic_id']},{row['dialogue_id']},{row['speaker']}," \
                      f"{row['utterance']},{pron},{trans}\n"
        f.write(result_line)
        f.flush()
        time.sleep(0.8)

# ▶️ 전체 결과 병합 저장
df = pd.read_csv(SAVE_PATH, header=None,
                 names=["topic_id", "dialogue_id", "speaker", "utterance", "pronunciation", "translation"])
df.to_csv(FINAL_PATH, index=False, encoding="utf-8-sig")
print("✅ 저장 완료 →", FINAL_PATH)

driver.quit()
