from IPython.display import display, Markdown, HTML
import ipywidgets as widgets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 학습 및 평가 함수
def train_and_evaluate_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, report

# 결과 시각화 함수
def display_metrics(acc, report_dict, model_name, feature_names):
    display(Markdown(f"### 📊 선택한 모델: **{model_name}**"))
    display(Markdown(f"### 🔎 사용한 변수: {', '.join([f'`{v}`' for v in feature_names])}"))

    display(Markdown("## ✅ **모델 정확도**"))
    display(HTML(f"<div style='font-size: 24px; font-weight: bold; color: green;'>📈 정확도: {acc*100:.2f}%</div>"))
    display(HTML("<p style='font-size: 14px;'>✔️ 정확도는 전체 데이터 중 모델이 올바르게 예측한 비율입니다.</p>"))

    display(Markdown("## 🔍 클래스별 설명 및 성능"))

    precision_1 = report_dict['1']['precision']
    precision_0 = report_dict['0']['precision']

    table_html = f"""
    <table style="border-collapse: collapse; width: 100%;">
        <tr>
            <th style="border:1px solid #ccc; padding:8px;">산불 유형</th>
            <th style="border:1px solid #ccc; padding:8px;">클래스</th>
            <th style="border:1px solid #ccc; padding:8px;">정확도</th>
        </tr>
        <tr>
            <td style="border:1px solid #ccc; padding:8px;">🔥 대형 산불</td>
            <td style="border:1px solid #ccc; padding:8px;">1</td>
            <td style="border:1px solid #ccc; padding:8px;">{precision_1:.2f}</td>
        </tr>
        <tr>
            <td style="border:1px solid #ccc; padding:8px;">🌿 일반 산불</td>
            <td style="border:1px solid #ccc; padding:8px;">0</td>
            <td style="border:1px solid #ccc; padding:8px;">{precision_0:.2f}</td>
        </tr>
    </table>
    """
    display(HTML(table_html))

# 학습 실행 UI 함수
def run_training_ui(model_fn, df, feature_names):
    output = widgets.Output()
    run_btn = widgets.Button(description="🚀 모델 학습 시작", button_style='success')

    def on_run_click(b):
        with output:
            output.clear_output()
            display(Markdown("### ⏳ 모델을 학습하고 있어요..."))
            try:
                model = model_fn()
                model_name = type(model).__name__
                X = df[feature_names]
                y = df["large_fire"]
                acc, report = train_and_evaluate_model(model, X, y)
                display(Markdown("### 🎉 모델 학습 완료! 결과는 다음과 같습니다."))
                display_metrics(acc, report, model_name, feature_names)
            except Exception as e:
                display(Markdown(f"❌ 오류 발생: `{str(e)}`"))

    run_btn.on_click(on_run_click)
    display(widgets.VBox([run_btn, output]))
