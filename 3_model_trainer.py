import ipywidgets as widgets
from IPython.display import display, Markdown, HTML
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
