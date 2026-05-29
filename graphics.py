# Посмотрим на распределение нашей целевой переменной (target)
plt.figure(figsize=(8, 5))
sns.histplot(data['burnout_risk_score'], bins=50)

plt.title('Распределение риска выгорания')
plt.xlabel('Риск выгорания')
plt.ylabel('Количество сотрудников')
plt.show()

# Посмотрим на матрицу корреляции признаков
plt.figure(figsize=(12, 8))
# Отбираем только числовые колонки для корреляции
numeric_data = data.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Матрица корреляции признаков')
plt.show()

# Посмотрим на истинную важность признаков
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Находим средние и дисперсии на train и сразу применяем к train
X_train_scaled = scaler.fit_transform(X_train_num)
# На test мы ТОЛЬКО применяем (не пересчитываем средние!)
X_test_scaled = scaler.transform(X_test_num)

# Обучаем новую модель на отмасштабированных данных
model_scaled = Ridge()
model_scaled.fit(X_train_scaled, y_train)

weights_scaled = pd.Series(model_scaled.coef_, index=numeric_features)
weights_scaled.sort_values(ascending=False).plot(kind='bar', figsize=(10, 5), color='#1E90FF')
plt.title("Истинная важность признаков (после StandardScaler)")
plt.show()

# Иногда полезно посмотреть, на каких примерах модель ошибается сильнее всего. Построим гистограмму квадратов ошибок.
# Получаем предсказания на обучающей выборке
y_train_pred = pipeline.predict(X_train)
errors = (y_train - y_train_pred) ** 2

plt.figure(figsize=(8, 5))
sns.histplot(errors, bins=100)
plt.title('Распределение квадратов ошибок на обучающей выборке')
plt.show()
