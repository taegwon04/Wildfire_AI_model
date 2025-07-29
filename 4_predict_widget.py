import ipywidgets as widgets
from IPython.display import display, Markdown, HTML
import pandas as pd

# 입력 스타일 위젯 만들기
def choose_input_output_style():
    input_options = ["슬라이더", "숫자 입력", "드롭다운"]
    output_options = ["게이지바", "문장 피드백", "이모지", "색상 경고"]

    input_style = widgets.ToggleButtons(
        options=input_options,
        description="입력 방식:",
        button_style=""
    )
    output_style = widgets.ToggleButtons(
        options=output_options,
        description="출력 방식:",
        button_style=""
    )
    return input_style, output_style

# 입력 위젯 생성 함수
def create_input_widgets(variables, input_style, df):
    widgets_dict = {}
    for var in variables:
        min_val = float(df[var].min())
        max_val = float(df[var].max())
        mean_val = float(df[var].mean())

        if input_style == "슬라이더":
            w = widgets.FloatSlider(description=var, min=min_val, max=max_val, step=0.1, value=mean_val)
        elif input_style == "숫자 입력":
            w = widgets.FloatText(description=var, value=mean_val)
        elif input_style == "드롭다운":
            options = [round(x, 1) for x in list(df[var].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).unique())]
            w = widgets.Dropdown(description=var, options=options, value=mean_val)
        widgets_dict[var] = w
    return widgets_dict

# 출력 스타일별 결과 표시
def show_prediction(prob, output_style):
    percent = prob * 100
    if output_style == "게이지바":
        bar = f"""
        <div style="border:1px solid #ccc; width:400px; height:25px; background-color:#eee;">
            <div style="width:{percent}%; height:100%; background:linear-gradient(to right, #ff9999, #ff0000);"></div>
        </div>
        """
        display(HTML(f"<h3>🔥 산불 예측 확률: <b>{percent:.1f}%</b></h3>"))
        display(HTML(bar))
    elif output_style == "문장 피드백":
        if prob > 0.7:
            msg = "⚠️ 대형 산불 발생 확률이 매우 높습니다!"
        elif prob > 0.4:
            msg = "🔶 중간 수준의 위험이 감지됩니다."
        else:
            msg = "🟢 위험도 낮음: 안전합니다."
        display(Markdown(f"### 예측 결과: **{msg}**"))
    elif output_style == "이모지":
        if prob > 0.7:
            emoji = "🔥🔥🔥"
        elif prob > 0.4:
            emoji = "🔥🔥"
        else:
            emoji = "🌱"
        display(Markdown(f"## 예측 결과: {emoji}"))
    elif output_style == "색상 경고":
        color = "red" if prob > 0.7 else ("orange" if prob > 0.4 else "green")
        msg = f"<div style='padding:10px; background-color:{color}; color:white;'>산불 확률: {percent:.1f}%</div>"
        display(HTML(msg))

# 전체 위젯 UI
def prediction_ui(model, df, variables):
    input_style_widget, output_style_widget = choose_input_output_style()
    confirm_btn = widgets.Button(description="입력 시작", button_style='info')
    input_box = widgets.VBox()
    run_btn = widgets.Button(description="예측 실행", button_style='success')
    output_area = widgets.Output()

    def on_confirm(b):
        input_widgets = create_input_widgets(variables, input_style_widget.value, df)
        input_box.children = list(input_widgets.values())

        def on_click(run):
            output_area.clear_output()
            with output_area:
                values = [w.value for w in input_widgets.values()]
                df_input = pd.DataFrame([values], columns=variables)
                try:
                    prob = model.predict_proba(df_input)[0][1]
                    show_prediction(prob, output_style_widget.value)
                except Exception as e:
                    display(Markdown(f"❌ 오류 발생: `{str(e)}`"))

        run_btn.on_click(on_click)

    confirm_btn.on_click(on_confirm)

    display(widgets.VBox([
        Markdown("## 🎛️ 입력 및 출력 방식을 선택하세요"),
        input_style_widget,
        output_style_widget,
        confirm_btn,
        input_box,
        run_btn,
        output_area
    ]))
