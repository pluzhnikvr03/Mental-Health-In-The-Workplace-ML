import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%config InlineBackend.figure_format = 'retina'

# Настройки для светлого фона
plt.style.use('seaborn-v0_8-whitegrid')  # или 'seaborn-v0_8-darkgrid'
sns.set_theme(style='whitegrid',
              rc={'axes.facecolor': 'white',
                  'figure.facecolor': 'white',
                  'text.color': 'black',
                  'axes.labelcolor': 'black',
                  'xtick.color': 'black',
                  'ytick.color': 'black'})

import warnings
warnings.filterwarnings('ignore')

# Список категориальных колонок с пропусками
categorical_fillna = [
    'mental_health_condition',
    'employer_support_level',
    'used_eap',
    'workplace_stigma_felt'
]

# Заполняем Unknown
for col in categorical_fillna:
    if col in data.columns:
        data[col] = data[col].fillna('Unknown')

y = data['burnout_risk_score']

# Удаляем таргет и текстовые идентификаторы
drop_cols = [	'record_id',	'year', 'weekly_overtime_hours', 'annual_salary_usd', 'intention_to_leave', 'remote_work_preference', 'burnout_risk_score']
X = data.drop(columns=drop_cols)

# Разобьем выборку на обучающую (train) и тестовую (test). Мы будем обучать модель на train, а проверять качество на test.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)

# Линейная регрессия принимает на вход только числа. Давайте для начала выделим только числовые признаки и обучим модель на них.
numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("Числовые признаки:", numeric_features)

X_train_num = X_train[numeric_features]
X_test_num = X_test[numeric_features]

# L2-регуляризация (Ridge, гребневая регуляция) — это метод в машинном обучении, который предотвращает переобучение модели, добавляя к функции потерь штраф, пропорциональный квадрату суммы весов.
from sklearn.linear_model import Ridge

# Создаем объект модели
model = Ridge()

# Обучаем модель (подбираем веса w и w0)
model.fit(X_train_num, y_train)

# Модель обучилась! Теперь мы можем делать предсказания.
y_pred = model.predict(X_test_num)
y_pred[:5]

# Посчитаем качество. Основная метрика для задачи регрессии — RMSE (Root Mean Squared Error).
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE на тесте: {rmse:.2f}")

# Для понимания правильности нам нужен бейзлайн — простейшая модель. Например, если мы всегда будем предсказывать просто среднее выгорание сотрудников из обучающей выборки.
y_pred_baseline = np.full_like(y_test, y_train.mean())
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
print(f"RMSE бейзлайна (просто среднее): {rmse_baseline:.2f}")

# Давайте посмотрим на веса, которые выучила модель. Чем больше модуль веса, тем сильнее признак влияет на популярность.
weights = pd.Series(model.coef_, index=numeric_features)
weights.sort_values(ascending=False)

# Чтобы честно сравнивать веса, признаки нужно отмасштабировать (стандартизировать).
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

# У модели Ridge есть гиперпараметр alpha — сила регуляризации. Подберем его оптимальное значение с помощью GridSearchCV.
from sklearn.model_selection import GridSearchCV

# Задаем сетку гиперпараметров
param_grid = {'alpha': np.logspace(-3, 3, 20)}

# cv=5 означает 5-фолдовую кросс-валидацию
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

print(f"Лучшее значение alpha: {grid_search.best_params_['alpha']}")
print(f"Лучший RMSE на кросс-валидации: {-grid_search.best_score_:.2f}")

# Мы использовали только числа. А ведь у нас есть категориальные признаки. Линейная модель не понимает строк. Нам нужно применить One-Hot Encoding.
# Чтобы не применять Scaler к одним колонкам руками, а OHE к другим, в sklearn есть ColumnTransformer.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

categorical_features = ['country',	'industry', 'job_role', 'employment_type', 'work_model', 'company_size', 'age_group', 'gender', 'mental_health_condition', 'has_diagnosis', 'treatment_type', 'stress_level', 'employer_support_level', 'mental_health_policy_exists', 'eap_available', 'used_eap',	'workplace_stigma_felt']

# Создаем трансформер: к числам применяем StandardScaler, к категориям — OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Собираем всё в Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=grid_search.best_params_['alpha']))
])

# Обучаем ВЕСЬ пайплайн одной строчкой!
pipeline.fit(X_train, y_train)

# Предсказываем
y_pred_pipe = pipeline.predict(X_test)
rmse_pipe = np.sqrt(mean_squared_error(y_test, y_pred_pipe))
print(f"RMSE с категориальными признаками: {rmse_pipe:.2f}")

# Lasso(L1-регуляризация0 умеет делать отбор признаков, зануляя веса у бесполезных. После OneHotEncoding у нас стало очень много признаков (каждый поджанр стал отдельной колонкой). Посмотрим, сможет ли Lasso выкинуть лишнее.
from sklearn.linear_model import Lasso

pipeline_lasso = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Lasso(alpha=0.1)) # alpha=0.1 для Лассо обычно достаточно сильно режет
])

pipeline_lasso.fit(X_train, y_train)

# Считаем количество нулевых весов
zero_weights = np.sum(pipeline_lasso.named_steps['model'].coef_ == 0)
total_weights = len(pipeline_lasso.named_steps['model'].coef_)

print(f"Lasso занулил {zero_weights} признаков из {total_weights}!")
rmse_lasso = np.sqrt(mean_squared_error(y_test, pipeline_lasso.predict(X_test)))
print(f"RMSE Lasso: {rmse_lasso:.2f}")

# Анализ ошибок (Residuals)
# Иногда полезно посмотреть, на каких примерах модель ошибается сильнее всего. Построим гистограмму квадратов ошибок.

# Получаем предсказания на обучающей выборке
y_train_pred = pipeline.predict(X_train)
errors = (y_train - y_train_pred) ** 2

plt.figure(figsize=(8, 5))
sns.histplot(errors, bins=100)
plt.title('Распределение квадратов ошибок на обучающей выборке')
plt.show()

# У модели есть огромные ошибки (длинный хвост). Это "выбросы". Попробуем выкинуть топ-5% самых больших ошибок и переобучить модель.

# Берем 95-й квантиль ошибки
threshold = np.quantile(errors, 0.95)
mask = errors < threshold # Маска "хороших" треков

# Оставляем только те треки, где ошибка была не слишком большой
X_train_clean = X_train[mask]
y_train_clean = y_train[mask]

# Переобучаем
pipeline.fit(X_train_clean, y_train_clean)
y_pred_clean = pipeline.predict(X_test)
rmse_clean = np.sqrt(mean_squared_error(y_test, y_pred_clean))
print(f"RMSE после удаления выбросов: {rmse_clean:.2f}")

# Качество на тестовой выборке улучшилось! Удаление шума из обучающей выборки часто помогает линейным моделям.
