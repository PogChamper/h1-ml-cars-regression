import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re

st.set_page_config(
    page_title='Оценка стоимости авто',
    layout='wide'
)


@st.cache_resource
def load_model():
    with open('models/best_model_pipeline.pickle', 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv'
    return pd.read_csv(url)


def clean_numeric_column(series):
    '''Убирает единицы измерения и конвертирует в float'''
    return pd.to_numeric(series.str.extract(r'([\d.]+)')[0], errors='coerce')


def parse_torque_value(val):
    '''Парсит значение torque, возвращает (torque_nm, rpm)'''
    if pd.isna(val):
        return np.nan, np.nan
    
    val = str(val).upper()
    
    torque_match = re.search(r'([\d.]+)\s*(NM|KGM|@)?', val)
    if torque_match:
        torque_num = float(torque_match.group(1))
        if 'KGM' in val:
            torque_num *= 9.80665
    else:
        torque_num = np.nan
    
    rpm_match = re.search(r'@\s*([\d,]+)', val)
    rpm_num = float(rpm_match.group(1).replace(',', '')) if rpm_match else np.nan
    
    return torque_num, rpm_num


def preprocess_input(df, pipeline_data):
    '''Препроцессинг входных данных'''
    df = df.copy()
    
    df['mileage'] = clean_numeric_column(df['mileage'])
    df['engine'] = clean_numeric_column(df['engine'])
    df['max_power'] = clean_numeric_column(df['max_power'])
    
    torque_parsed = [parse_torque_value(val) for val in df['torque']]
    df['torque'] = [t[0] for t in torque_parsed]
    df['max_torque_rpm'] = [t[1] for t in torque_parsed]
    
    medians = {
        'mileage': 19.3, 'engine': 1248.0, 'max_power': 82.0,
        'torque': 113.0, 'seats': 5.0, 'max_torque_rpm': 3500.0
    }
    for col, median in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median)
    
    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)
    
    # Feature Engineering
    df['power_per_liter'] = df['max_power'] / (df['engine'] / 1000)
    df['year_squared'] = df['year'] ** 2
    df['age'] = 2024 - df['year']
    df['km_per_year'] = df['km_driven'] / df['age'].replace(0, 1)
    df['torque_per_liter'] = df['torque'] / (df['engine'] / 1000)
    df['log_km'] = np.log1p(df['km_driven'])
    
    df['brand'] = df['name'].str.split().str[0]
    df['model'] = df['name'].str.split().str[1].fillna('Unknown')
    
    owner_map = pipeline_data['owner_map']
    df['owner_num'] = df['owner'].map(owner_map).fillna(1)
    df['is_third_plus_owner'] = (df['owner_num'] >= 3).astype(int)
    df['is_premium_seller'] = ((df['owner_num'] <= 2) & (df['seller_type'] == 'Dealer')).astype(int)
    
    median_km = pipeline_data['median_km_per_year']
    df['is_low_mileage'] = (df['km_per_year'] < median_km).astype(int)
    df['is_auto_diesel'] = ((df['transmission'] == 'Automatic') & (df['fuel'] == 'Diesel')).astype(int)
    
    return df


def predict(df, pipeline_data):
    '''Предсказание цены'''
    num_cols = pipeline_data['num_cols_fe2']
    cat_cols = pipeline_data['cat_cols_fe2']
    
    X_cat = pipeline_data['ohe'].transform(df[cat_cols].astype(str))
    X_num = df[num_cols].values
    X_poly = pipeline_data['poly'].transform(X_num)
    X_full = np.hstack([X_poly, X_cat])
    X_scaled = pipeline_data['scaler'].transform(X_full)
    
    y_pred_log = pipeline_data['model'].predict(X_scaled)
    return np.expm1(y_pred_log)


def render_eda(df_train):
    '''Страница EDA'''
    st.header('Exploratory Data Analysis')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Распределение цен')
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_train['selling_price'], bins=50, ax=ax, color='steelblue')
        ax.set_xlabel('Цена')
        ax.set_ylabel('Количество')
        st.pyplot(fig)
    
    with col2:
        st.subheader('Цена по типу топлива')
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_train, x='fuel', y='selling_price', ax=ax)
        ax.set_xlabel('Тип топлива')
        ax.set_ylabel('Цена')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader('Цена vs Год выпуска')
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df_train.sample(1000), x='year', y='selling_price',
            hue='transmission', alpha=0.6, ax=ax
        )
        ax.set_xlabel('Год')
        ax.set_ylabel('Цена')
        st.pyplot(fig)
    
    with col4:
        st.subheader('Цена по трансмиссии')
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_train, x='transmission', y='selling_price', ax=ax)
        ax.set_xlabel('Трансмиссия')
        ax.set_ylabel('Цена')
        st.pyplot(fig)
    
    st.subheader('Корреляционная матрица')
    numeric_cols = ['year', 'selling_price', 'km_driven', 'seats']
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df_train[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, center=0)
    st.pyplot(fig)
    
    st.subheader('Статистика данных')
    st.dataframe(df_train.describe())


def render_prediction(pipeline_data):
    '''Страница предсказания'''
    st.header('Предсказание стоимости автомобиля')
    
    if pipeline_data is None:
        st.error('Модель не загружена')
        return
    
    input_method = st.radio('Способ ввода данных:', ['Ручной ввод', 'Загрузка CSV'])
    
    if input_method == 'Ручной ввод':
        col1, col2, col3 = st.columns(3)
        
        with col1:
            name = st.text_input('Название авто', 'Maruti Swift VXI')
            year = st.number_input('Год выпуска', 2000, 2024, 2015)
            km_driven = st.number_input('Пробег (км)', 0, 1000000, 50000)
            fuel = st.selectbox('Топливо', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
        
        with col2:
            seller_type = st.selectbox('Тип продавца', ['Individual', 'Dealer', 'Trustmark Dealer'])
            transmission = st.selectbox('Трансмиссия', ['Manual', 'Automatic'])
            owner = st.selectbox('Владелец', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
            seats = st.number_input('Количество мест', 2, 14, 5)
        
        with col3:
            mileage = st.text_input('Расход (kmpl)', '18.0 kmpl')
            engine = st.text_input('Объем двигателя (CC)', '1197 CC')
            max_power = st.text_input('Мощность (bhp)', '82 bhp')
            torque = st.text_input('Крутящий момент', '113Nm@ 4200rpm')
        
        if st.button('Предсказать цену', type='primary'):
            input_data = pd.DataFrame([{
                'name': name, 'year': year, 'km_driven': km_driven, 'fuel': fuel,
                'seller_type': seller_type, 'transmission': transmission, 'owner': owner,
                'mileage': mileage, 'engine': engine, 'max_power': max_power,
                'torque': torque, 'seats': seats
            }])
            
            try:
                processed = preprocess_input(input_data, pipeline_data)
                prediction = predict(processed, pipeline_data)[0]
                st.success(f'Предсказанная цена: {prediction:,.0f} рупий (~{prediction/83:,.0f} USD)')
            except Exception as e:
                st.error(f'Ошибка: {e}')
    
    else:
        uploaded_file = st.file_uploader('Загрузите CSV файл', type='csv')
        
        if uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)
            st.write('Загруженные данные:')
            st.dataframe(df_input.head())
            
            if st.button('Предсказать цены', type='primary'):
                try:
                    processed = preprocess_input(df_input, pipeline_data)
                    df_input['predicted_price'] = predict(processed, pipeline_data)
                    
                    st.write('Результаты:')
                    st.dataframe(df_input[['name', 'year', 'km_driven', 'predicted_price']])
                    
                    csv = df_input.to_csv(index=False)
                    st.download_button('Скачать результаты', csv, 'predictions.csv', 'text/csv')
                except Exception as e:
                    st.error(f'Ошибка: {e}')


def render_weights(pipeline_data):
    '''Страница весов модели'''
    st.header('Визуализация весов модели')
    
    if pipeline_data is None:
        st.error('Модель не загружена')
        return
    
    st.subheader('Информация о модели')
    st.write(f"Описание: {pipeline_data['description']}")
    st.write(f"Test R2: {pipeline_data['test_r2']:.4f}")
    st.write(f"Best alpha: {pipeline_data['best_alpha']}")
    
    model = pipeline_data['model']
    coefs = model.coef_
    
    st.subheader('Распределение коэффициентов')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(coefs, bins=50, ax=ax, color='steelblue')
    ax.set_xlabel('Значение коэффициента')
    ax.set_ylabel('Количество')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    st.subheader('Топ-20 коэффициентов по модулю')
    
    num_features = pipeline_data['num_cols_fe2']
    poly = pipeline_data['poly']
    ohe = pipeline_data['ohe']
    
    poly_names = poly.get_feature_names_out(num_features)
    ohe_names = ohe.get_feature_names_out()
    all_names = list(poly_names) + list(ohe_names)
    
    coef_df = pd.DataFrame({
        'feature': all_names[:len(coefs)],
        'coefficient': coefs,
        'abs_coef': np.abs(coefs)
    }).sort_values('abs_coef', ascending=False)
    
    top_20 = coef_df.head(20)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['green' if c > 0 else 'red' for c in top_20['coefficient']]
    ax.barh(range(len(top_20)), top_20['coefficient'], color=colors)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['feature'])
    ax.set_xlabel('Коэффициент')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    
    st.caption('Зеленый = положительное влияние | Красный = отрицательное влияние')
    
    with st.expander('Показать все коэффициенты'):
        st.dataframe(coef_df)


def main():
    st.title('Оценка стоимости подержанных автомобилей')
    st.markdown('---')
    
    page = st.sidebar.selectbox(
        'Раздел',
        ['EDA', 'Предсказание', 'Веса модели']
    )
    
    try:
        pipeline_data = load_model()
    except Exception:
        pipeline_data = None
        st.sidebar.warning('Модель не загружена. Запустите ячейку сохранения в ноутбуке.')
    
    df_train = load_data()
    
    if page == 'EDA':
        render_eda(df_train)
    elif page == 'Предсказание':
        render_prediction(pipeline_data)
    elif page == 'Веса модели':
        render_weights(pipeline_data)


if __name__ == '__main__':
    main()
