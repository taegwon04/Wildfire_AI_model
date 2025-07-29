import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML

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
    plt.title(f"📊 {var_name} ", fontsize=14)
    plt.xlabel(var_name)
    plt.ylabel("frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

