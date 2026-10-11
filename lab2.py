import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import (
    mean_squared_error, root_mean_squared_error, mean_absolute_error,
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,
    classification_report,
)
plt.rcParams["figure.dpi"] = 100

df = pd.read_csv("train.csv")
print("=" * 60)
print("ДАТАСЕТ:", df.shape)
print(df.head(3))
print(df.info())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]
print("\nПропуски после заполнения:", int(df.isna().sum().sum()))

print("\n" + "=" * 60)
print("ЗАДАЧА РЕГРЕССИИ: 预测 возраст Age")
print("=" * 60)
TARGET_REG = "Age"
X_reg = df.drop(DROP_COLS + [TARGET_REG, "Survived"], axis=1)
y_reg = df[TARGET_REG]

X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.4, random_state=42)
X_te, X_va, y_te, y_va = train_test_split(X_te, y_te, test_size=0.4, random_state=42)
print("Размеры — train:", X_tr.shape[0], "| test:", X_te.shape[0], "| val:", X_va.shape[0])

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
X_va_s = scaler.transform(X_va)

lin = LinearRegression()
lin.fit(X_tr_s, y_tr)
y_pred_lin = lin.predict(X_te_s)

def report_reg(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"[{name}] MSE={mse:.2f} | RMSE={rmse:.2f} | MAE={mae:.2f}")

print("\n-- Оценка моделей на тестовой выборке --")
report_reg("LinearRegression", y_te, y_pred_lin)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_tr_p = poly.fit_transform(X_tr_s)
X_te_p = poly.transform(X_te_s)
poly_lin = LinearRegression()
poly_lin.fit(X_tr_p, y_tr)
y_pred_poly = poly_lin.predict(X_te_p)
report_reg("Polynomial deg=2", y_te, y_pred_poly)

ridge = Ridge(alpha=5.0)
ridge.fit(X_tr_s, y_tr)
y_pred_ridge = ridge.predict(X_te_s)
report_reg("Ridge (L2)", y_te, y_pred_ridge)

scores = {
    "Linear": root_mean_squared_error(y_te, y_pred_lin),
    "Poly(2)": root_mean_squared_error(y_te, y_pred_poly),
    "Ridge": root_mean_squared_error(y_te, y_pred_ridge),
}
best_reg = min(scores, key=scores.get)
print("\nЛучшая регрессионная модель:", best_reg,
      f"(RMSE={scores[best_reg]:.2f})")

print("\n" + "=" * 60)
print("ЗАДАЧА КЛАССИФИКАЦИИ: 预测 выживание Survived (0/1)")
print("=" * 60)
TARGET_CLS = "Survived"
X_cls = df.drop(DROP_COLS + [TARGET_CLS], axis=1)
y_cls = df[TARGET_CLS]

X_ctr, X_cte, y_ctr, y_cte = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)
scaler_cls = StandardScaler()
X_ctr_s = scaler_cls.fit_transform(X_ctr)
X_cte_s = scaler_cls.transform(X_cte)

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_ctr_s, y_ctr)
y_pred_cls = logreg.predict(X_cte_s)

acc = accuracy_score(y_cte, y_pred_cls)
prec = precision_score(y_cte, y_pred_cls)
rec = recall_score(y_cte, y_pred_cls)
f1 = f1_score(y_cte, y_pred_cls)
cm = confusion_matrix(y_cte, y_pred_cls)

print("\n-- Оценка классификации --")
print(f"Accuracy = {acc:.3f}")
print(f"Precision = {prec:.3f}")
print(f"Recall = {rec:.3f}")
print(f"F1-score = {f1:.3f}")

print("\nConfusion Matrix:")
print("        [Pred 0 Pred 1]")
print(" True 0 ", cm[0], "  (TN, FP)")
print(" True 1 ", cm[1], "  (FN, TP)")

print("\nClassification report:")
print(classification_report(y_cte, y_pred_cls))

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="bwr",
            xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("\nСохранено изображение: confusion_matrix.png")

print("\n" + "=" * 60)
print("ВЫГРУЗКА НА GITHUB:")
print("""
  cd /путь/к/проекту
  git init
  git add .
  git commit -m "Лабораторная работа №2 Регрессия и классификация"
  git branch -M main
  git remote add origin https://github.com/ВАШ_ЛОГИН/название_репо.git
  git push -u origin main
""")
print("Лабораторная работа №2 выполнена.")
