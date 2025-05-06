import os
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ========================
#  경로 설정
# ========================
orange_path = '../8th_Team_data/orange fruit'
not_orange_base = '../8th_Team_data/not_orange_fruits'
not_orange_subfolders = ['strawberry', 'apple', 'tangerine']
allowed_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')

# 데이터를 담을 리스트
image_data = []
labels = []

# ========================
#  이미지 전처리 함수
# ========================
def imread_unicode(path):
    """
    한글 등 유니코드 경로에서 안전하게 이미지를 읽기 위한 함수
    """
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)

def augment_image(img):
    """
    1) 원본 이미지
    2) 좌우 반전
    3) 90도 회전
    4) 180도 회전
    5) 270도 회전
    총 5가지 이미지를 반환
    """
    imgs = []
    
    # 원본
    imgs.append(img)
    # 좌우 반전
    imgs.append(cv2.flip(img, 1))
    # 90도 회전
    imgs.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    # 180도 회전
    imgs.append(cv2.rotate(img, cv2.ROTATE_180))
    # 270도 회전
    imgs.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    
    return imgs

def load_images_from_folder(folder_path, label_name):
    """
    폴더 경로 내 이미지를 읽고, augment_image로 생성된 여러 버전을
    64×64 사이즈로 리사이즈하여 image_data 및 labels 리스트에 추가.
    """
    loaded = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(allowed_exts):
            path = os.path.join(folder_path, filename)
            img = imread_unicode(path)
            if img is None:
                print(f"[❌ FAIL] 이미지 로드 실패: {path}")
                continue
            try:
                # 증강된 이미지들에 대해 리사이즈 후 데이터셋에 추가
                for aug in augment_image(img):
                    resized = cv2.resize(aug, (64, 64))
                    flat = resized.flatten()
                    image_data.append(flat)
                    labels.append(label_name)
                    loaded += 1
            except Exception as e:
                print(f"[ERROR] {path} 처리 중 오류: {e}")
    print(f"[✅ DONE] {label_name} 증강 포함 {loaded}장 처리됨.")

# ========================
#  메인 프로세스
# ========================
def main():
    # -----------------------
    # 1) 이미지 로딩 & 증강
    # -----------------------
    load_images_from_folder(orange_path, 'orange')
    for subfolder in not_orange_subfolders:
        load_images_from_folder(os.path.join(not_orange_base, subfolder), 'not_orange')

    print(f"총 이미지 수: {len(image_data)}")
    if len(image_data) == 0:
        print("[🚨 경고] image_data가 비어 있습니다. 전처리 실패!")
        return

    # -----------------------
    # 2) DataFrame 생성 & 저장
    # -----------------------
    df = pd.DataFrame(image_data)
    df['label'] = labels
    df.to_csv('orange_dataset_color.csv', index=False)
    print("[저장 완료] orange_dataset_color.csv")

    # -----------------------
    # 3) 학습 데이터 준비
    # -----------------------
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------
    # (선택) GridSearchCV를 사용한 파라미터 튜닝
    # -----------------------
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }

    model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("\n[GRID SEARCH] Best Params:", grid_search.best_params_)
    print("[GRID SEARCH] Best Score (F1_macro):", grid_search.best_score_)

    # -----------------------
    # 4) 최적 모델로 훈련
    # -----------------------
    best_model.fit(X_train, y_train)

    # -----------------------
    # 5) 모델 저장
    # -----------------------
    joblib.dump(best_model, 'decision_tree_model.pkl')
    print("[저장 완료] rf_model_tree.pkl")

    # -----------------------
    # 6) 예측 및 평가
    # -----------------------
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    print("\n[훈련 정확도]", accuracy_score(y_train, y_train_pred))
    print("[테스트 정확도]", accuracy_score(y_test, y_test_pred))
    print("\n", classification_report(y_test, y_test_pred))

    # -----------------------
    # 7) 새로운 이미지 테스트
    # -----------------------
    def predict_new_image(image_path):
        img = imread_unicode(image_path)
        if img is not None:
            # 예시: 180도 회전 + 좌우 반전
            rotated = cv2.rotate(img, cv2.ROTATE_180)
            flipped = cv2.flip(rotated, 1)
            resized = cv2.resize(flipped, (64, 64))
            flat = resized.flatten().reshape(1, -1)
            prediction = best_model.predict(flat)[0]
            print(f"[예측 결과] '{image_path}' 는 '{prediction}'으로 분류되었습니다.")
        else:
            print(f"[ERROR] 이미지 로드 실패: {image_path}")

    # 테스트할 이미지 경로 지정 (예: 오렌지)
    test_image_path = '../8th_Team_data/not_orange_fruits/apple/Apple_501.jpg'
    predict_new_image(test_image_path)

# ========================
#  메인 실행
# ========================
if __name__ == "__main__":
    main()
