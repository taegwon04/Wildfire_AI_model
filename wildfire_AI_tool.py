from IPython.display import display, Markdown, HTML
import ipywidgets as widgets

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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
