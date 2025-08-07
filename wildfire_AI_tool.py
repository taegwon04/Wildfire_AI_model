from IPython.display import display, Markdown, HTML
from IPython.display import clear_output
import ipywidgets as widgets

import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1-----------------------------------------------------------------------------------------------------------------------------------------------------------
# 변수 설명 사전
explanations = {
    "기온": "🌡️ **기온**: 7일 전의 평균 기온 (℃)",
    "습도": "💧 **습도**: 7일 전의 평균 습도 (%)",
    "풍속": "💨 **풍속**: 7일 전의 풍속 (km/h)",
    "강수량": "🌧️ **강수량**: 7일 전의 강수량 (mm)",
    "고립도": "🌲 **고립도**: 산불 접근의 어려움 (0~1 사이 값) 값이 높을수록 접근이 어려움",
    "월": "🌕 **월**: 산불 발생 달"
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

# 모델 설명
model_descriptions = {
    "랜덤 포레스트": "🌳 **랜덤 포레스트**는 여러 개의 나무(결정 트리)가 모여서 투표하듯이 예측하는 모델이에요. 정확하고 튼튼한 결과를 잘 만들어냅니다.",
    "로지스틱 회귀": "📈 **로지스틱 회귀**는 데이터를 기준으로 직선을 하나 그어 각 데이터가 어느 쪽에 속할 확률을 계산해서 결정하는 분류하는 간단한 모델이에요. 계산이 빠르고 직관적입니다.",
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

    # 모델 선택 UI 묶음
    container = widgets.VBox(
        [model_toggle, desc_box],
        layout=widgets.Layout(
            padding='20px',
            border='2px solid #2a5298', # 예측 프로그램 빼고 이 색으로 통일일
            border_radius='12px',
            background_color='#1e3c72'
        )
    )

    display(container)

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

def run_training_ui(model_fn, df, feature_names):
    output = widgets.Output()

    # 월 변수 가중치 슬라이더
    month_weight_slider = widgets.FloatSlider(
        value=1.0, min=0.0, max=3.0, step=0.1,
        description="📅 월 변수 가중치", continuous_update=False,
        layout=widgets.Layout(width="400px")
    )

    # 실행 버튼
    run_btn = widgets.Button(description="🚀 모델 학습 시작", button_style='success')

    def on_run_click(b):
        with output:
            output.clear_output()
            display(Markdown("### 🔄 모델을 학습하고 있습니다..."))
            try:
                model = model_fn()
                df_copy = df.copy()

                df_copy["Month_weighted"] = df_copy["월"] * month_weight_slider.value
                final_features = feature_names + ["Month_weighted"]

                X = df_copy[final_features]
                y = df_copy["대형산불"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                display(Markdown(f"## ✅ 정확도: **{acc*100:.2f}%**"))
                table_html = "<table><tr><th>클래스</th><th>정확도</th></tr>"
                label_names = {"0": "비대형 산불 (0)", "1": "대형 산불 (1)"}
                for label in ["0", "1"]:
                    label_text = label_names[label]
                    precision = report[label]["precision"]
                    table_html += f"<tr><td>{label_text}</td><td>{precision:.2f}</td></tr>"
                table_html += "</table>"

                display(HTML(table_html))
            except Exception as e:
                display(Markdown(f"❌ 오류 발생: {str(e)}"))

    run_btn.on_click(on_run_click)

    # 인터랙티브 위젯 묶음 컨테이너너
    training_box = widgets.VBox(
        [month_weight_slider, run_btn, output],
        layout=widgets.Layout(
            padding='20px',
            border='2px solid #2a5298',
            border_radius='12px',
            background_color='#1e3c72'
        )
    )

    # 컨테이너 표시시
    display(training_box)


# 4-----------------------------------------------------------------------------------------------------------------------------------------------------------

# 입력 위젯 생성기
def create_input_widgets(style, variables, slider_ranges):
    widgets_dict = {}
    for var in variables:
        min_val, max_val, step = slider_ranges.get(var, (0, 100, 1))
        if style == "슬라이더":
            widgets_dict[var] = widgets.FloatSlider(
                value=(min_val + max_val) / 2,
                min=min_val, max=max_val, step=step,
                description=var, continuous_update=False, layout=widgets.Layout(width="400px")
            )
        elif style == "숫자 입력":
            widgets_dict[var] = widgets.BoundedFloatText(
                value=(min_val + max_val) / 2,
                min=min_val, max=max_val, step=step,
                description=var, layout=widgets.Layout(width="250px")
            )
        elif style == "드롭다운":
            values = [round(min_val + i * step, 2) for i in range(int((max_val - min_val) / step) + 1)]
            widgets_dict[var] = widgets.Dropdown(
                options=values, description=var, layout=widgets.Layout(width="250px")
            )
    return widgets_dict

# 출력 스타일 함수들
def show_result_gauge(prob):
    bar_html = f"""
    <div style='border:1px solid #ccc; width:400px; height:25px; background-color:#eee;'>
        <div style='width:{prob*100:.1f}%; height:100%; background:linear-gradient(to right, #ffff66, #ff3300);'></div>
    </div>
    """
    display(HTML(bar_html))

def show_result_sentence(prob):
    if prob >= 0.7:
        display(Markdown("🔥 **주의: 산불 발생 가능성이 매우 높습니다!**"))
    elif prob >= 0.4:
        display(Markdown("⚠️ **경계: 산불 가능성이 중간 정도입니다.**"))
    else:
        display(Markdown("🌱 **안전: 산불 발생 가능성은 낮습니다.**"))

def show_result_emoji(prob):
    emoji = "🟢"
    if prob >= 0.7:
        emoji = "🔴"
    elif prob >= 0.4:
        emoji = "🟡"
    display(Markdown(f"### 대형 산불 발생 확률: {emoji} ({prob*100:.1f}%)"))

def show_result_color(prob):
    # 예측 수준별 색상과 문구
    if prob < 0.4:
        color = "#28a745"  # 초록
        message = "🌱 안전: 산불 가능성 낮음"
    elif prob < 0.7:
        color = "#ffc107"  # 노랑
        message = "⚠️ 경고: 산불 가능성 중간"
    else:
        color = "#dc3545"  # 빨강
        message = "🔥 주의: 산불 가능성 매우 높음"

    # 강제 스타일 적용 및 캐시 무시를 위한 HTML 출력
    html_code = f"""
    <div style="
        background-color: {color};
        color: white;
        padding: 20px;
        margin: 10px 0;
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        transition: all 0.3s ease-in-out;
    ">
        {message} ({prob*100:.1f}%)
    </div>
    """
    clear_output(wait=True)  # 캐시된 출력 지우고 강제로 새로 출력
    display(HTML(html_code))

# 에측 프로그램
def prediction_widget_ui(model, variables, input_type, output_styles, month_weight):
    slider_ranges = {
        "기온": (0, 50, 0.5),
        "습도": (0, 100, 1),
        "풍속": (0, 20, 0.5),
        "강수량": (0, 300, 1),
        "고립도": (0, 1, 0.01),
        "월": (1, 12, 1)  # 월 선택 추가
    }

    # "월"이 포함되어 있지 않으면 강제로 추가
    if "월" not in variables:
        variables = variables + ["월"]

    input_widgets = create_input_widgets(input_type, variables, slider_ranges)
    output = widgets.Output()
    button = widgets.Button( description="🚀 예측 실행",layout=widgets.Layout(width='200px'),style=widgets.ButtonStyle(button_color='#ff9800'))  # 불꽃 주황색

    def on_click(b):
        with output:
            output.clear_output()
            values = [input_widgets[var].value for var in variables]

            # 월 처리
            month_value = input_widgets["월"].value
            month_weighted = month_value * month_weight

            # '월' 제거하고 Month_weighted만 입력 데이터에 포함
            input_data = {var: input_widgets[var].value for var in variables if var != "월"}
            input_data["Month_weighted"] = month_weighted

            df_input = pd.DataFrame([input_data])
            prob = model.predict_proba(df_input)[0][1]

            display(Markdown(f"## 🔍 대형 산불 발생 확률: **{prob*100:.1f}%**"))

            for style in output_styles:
                if style == "게이지바":
                    show_result_gauge(prob)
                elif style == "문장 피드백":
                    show_result_sentence(prob)
                elif style == "이모지":
                    show_result_emoji(prob)
                elif style == "색상 경고":
                    show_result_color(prob)

    button.on_click(on_click)


    # UI 레이아웃
    container = widgets.VBox(
        [*input_widgets.values(), button, output],
        layout=widgets.Layout(
            padding='30px',
            background_color='#fff3cd',
            border='2px solid #fff3cd',
            border_radius='12px',
            width='100%',
        )
    )

    styled_ui = widgets.HTML(
        value="""
        <style>
            .custom-container {
                background: linear-gradient(to right, #ffcc00, #ff3300);
                padding: 30px;
                border-radius: 16px;
                color: white;
                font-family: 'Segoe UI', sans-serif;
                margin-bottom: 20px;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
            }
        </style>
        <div class="custom-container">
            <h2 style="margin-top: 0;">🔥 대형 산불 발생 예측 시뮬레이터</h2>
            <p>아래에서 변수들을 조절하고<br>
            대형 산불 확률을 예측해 보아요!</p>
        </div>
        """
    )
    display(widgets.VBox([styled_ui, container]))


