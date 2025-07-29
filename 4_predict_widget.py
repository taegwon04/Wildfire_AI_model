
import ipywidgets as widgets
from IPython.display import display, Markdown, HTML
import pandas as pd

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
    color = "#28a745" if prob < 0.4 else "#ffc107" if prob < 0.7 else "#dc3545"
    display(HTML(f"<div style='padding:10px; background-color:{color}; color:white;'>🔥 산불 가능성: {prob*100:.1f}%</div>"))

# 메인 UI 함수
def prediction_widget_ui(model, variables, input_type, output_styles):
    slider_ranges = {
        "Temp_pre_7": (0, 50, 0.5),
        "Hum_pre_7": (0, 100, 1),
        "Wind_pre_7": (0, 20, 0.5),
        "Prec_pre_7": (0, 300, 1),
        "remoteness": (0, 1, 0.01)
    }

    input_widgets = create_input_widgets(input_type, variables, slider_ranges)
    output = widgets.Output()
    button = widgets.Button(description="🚀 예측 실행", button_style="primary")

    def on_click(b):
        with output:
            output.clear_output()
            values = [input_widgets[var].value for var in variables]
            df_input = pd.DataFrame([values], columns=variables)
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
    display(widgets.VBox([*input_widgets.values(), button, output]))
