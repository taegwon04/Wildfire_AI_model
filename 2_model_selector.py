import ipywidgets as widgets
from IPython.display import display, Markdown
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

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
    model_radio = widgets.RadioButtons(
        options=list(model_options.keys()),
        description='🤖 모델 선택:',
        style={'description_width': '100px'}
    )

    desc_box = widgets.Output()

    def on_model_change(change):
        with desc_box:
            desc_box.clear_output()
            model_name = change['new']
            explanation = model_descriptions[model_name]
            display(Markdown(f"### 📘 모델 설명\n{explanation}"))

    model_radio.observe(on_model_change, names='value')

    # 초기 설명도 표시
    with desc_box:
        model_name = model_radio.value
        explanation = model_descriptions[model_name]
        display(Markdown(f"### 📘 모델 설명\n{explanation}"))

    display(widgets.VBox([model_radio, desc_box]))

    return lambda: model_options[model_radio.value]
