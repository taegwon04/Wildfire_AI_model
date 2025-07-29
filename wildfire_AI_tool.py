from IPython.display import display, Markdown, HTML
import ipywidgets as widgets

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1-----------------------------------------------------------------------------------------------------------------------------------------------------------
# 변수 설명 사전
explanations = {
    "Temp_pre_7": "🌡️ **Temp_pre_7**: 7일 전의 평균 기온 (℃)",
    "Hum_pre_7": "💧 **Hum_pre_7**: 7일 전의 평균 습도 (%)",
    "Wind_pre_7": "💨 **Wind_pre_7**: 7일 전의 풍속 (km/h)",
    "Prec_pre_7": "🌧️ **Prec_pre_7**: 7일 전의 강수량 (mm)",
    "remoteness": "🌲 **remoteness**: 산불 접근의 어려움 (0~1 사이 값)"
}

def show_variable_info(df, var_name):
    if var_name not in df.columns:
        display(Markdown(f"❌ 변수 `{var_name}` 이(가) 데이터프레임에 없습니다."))
        return

    # 변수 설명 출력
    explanation = explanations.get(var_name, f"`{var_name}` 변수에 대한 설명이 없습니다.")
    display(Markdown(f"""### ℹ️ 변수 정보  
{explanation}"""))

    # 분포 시각화
    plt.figure(figsize=(8, 4))
    sns.histplot(df[var_name].dropna(), kde=True, color="#3B82F6", edgecolor="black")
    plt.title(f"{var_name} ", fontsize=14)
    plt.xlabel(var_name)
    plt.ylabel("frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# 2-----------------------------------------------------------------------------------------------------------------------------------------------------------
# 모델 객체 정의
model_options = {
    "랜덤 포레스트": RandomForestClassifier(n_estimators=100),
    "로지스틱 회귀": LogisticRegression(max_iter=1000),
    "신경망 (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
}

# 학생 친화적인 설명
model_descriptions = {
    "랜덤 포레스트": "🌳 **랜덤 포레스트**는 여러 개의 나무(결정 트리)가 모여서 투표하듯이 예측하는 모델이에요. 정확하고 튼튼한 결과를 잘 만들어냅니다.",
    "로지스틱 회귀": "📈 **로지스틱 회귀**는 데이터를 기준으로 직선을 하나 그어 분류하는 간단한 모델이에요. 계산이 빠르고 직관적입니다.",
    "신경망 (MLP)": "🧠 **신경망(MLP)**은 사람의 뇌처럼 작동하는 구조로, 복잡한 패턴도 잘 찾아내는 모델이에요. 여러 층으로 연결된 뇌세포처럼 동작합니다."
}

# 모델 선택 UI
def model_selector_ui():
    model_toggle = widgets.ToggleButtons(
        options=list(model_options.keys()),
        description='🤖 모델 선택:',
        button_style='',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )

    desc_box = widgets.Output()

    def on_model_change(change):
        with desc_box:
            desc_box.clear_output()
            model_name = change['new']
            explanation = model_descriptions[model_name]
            display(Markdown(f"### 📘 모델 설명\n{explanation}"))

    model_toggle.observe(on_model_change, names='value')

    # 초기 설명도 표시
    with desc_box:
        model_name = model_toggle.value
        explanation = model_descriptions[model_name]
        display(Markdown(f"### 📘 모델 설명\n{explanation}"))

    display(widgets.VBox([model_toggle, desc_box]))

    return lambda: model_options[model_toggle.value]
# 3-----------------------------------------------------------------------------------------------------------------------------------------------------------

# 학습 및 평가 함수
def train_and_evaluate_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, report

# 결과 시각화 함수
def display_metrics(acc, report_dict):
    display(Markdown(f"## ✅ 정확도: **{acc*100:.2f}%**"))

    table_html = "<table style='border-collapse: collapse; width: 100%;'>"
    table_html += "<tr><th style='border:1px solid #ccc;'>클래스</th><th style='border:1px solid #ccc;'>정확도(예측한 것 중 진짜만 골라낸 비율)</th></tr>"
    label_names = {"0": "비대형 산불 (0)", "1": "대형 산불 (1)"}
    for label in ["0", "1"]:
        precision = report_dict[label]["precision"]
        label_text = label_names[label]
        table_html += f"<tr><td style='border:1px solid #ccc;'>{label_text}</td><td style='border:1px solid #ccc;'>{precision:.2f}</td></tr>"
    table_html += "</table>"

    display(HTML(table_html))

# 학습 실행 UI 함수
def run_training_ui(model_fn, df, feature_names):
    output = widgets.Output()

    # 실행 버튼
    run_btn = widgets.Button(description="🚀 모델 학습 시작", button_style='success')

    def on_run_click(b):
        with output:
            output.clear_output()
            display(Markdown("### 🔄 모델을 학습하고 있습니다..."))
            try:
                model = model_fn()
                X = df[feature_names]
                y = df["large_fire"]
                acc, report = train_and_evaluate_model(model, X, y)
                display(Markdown("### 🎉 모델 학습 완료! 결과는 다음과 같습니다."))
                display_metrics(acc, report)
            except Exception as e:
                display(Markdown(f"❌ 오류 발생: {str(e)}"))

    run_btn.on_click(on_run_click)
    display(widgets.VBox([run_btn, output]))
# 4-----------------------------------------------------------------------------------------------------------------------------------------------------------
