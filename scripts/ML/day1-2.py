import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import logging

# 设置日志，方便后续调试及记录信息
logging.basicConfig(level=logging.INFO)

def load_data():
    """读取数据并预处理目标列的格式"""
    X = pd.read_csv('../../data/Adult/X.csv')
    y = pd.read_csv('../../data/Adult/y.csv')
    # 数据清洗：去除首尾空格和末尾句点
    y_series = y.iloc[:, 0].str.strip().str.replace(r'\.$', '', regex=True)
    return X, y_series

def build_preprocessor():
    """构建预处理流水线，分别处理数值和分类特征"""
    # 数值特征预处理：填充缺失值并标准化
    numerical_features = ['age', 'fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # 分类特征预处理：填充缺失值，并进行OneHot编码
    categorical_features = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # 保留未指定的列
    )
    return preprocessor

def main():
    # 加载数据
    X, y = load_data()
    
    # 划分训练集和测试集，并保持DataFrame格式以保留列名
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train = pd.DataFrame(x_train, columns=X.columns)
    x_test = pd.DataFrame(x_test, columns=X.columns)
    
    # 替换分类变量中的缺失值符号 '?' 为 np.nan
    categorical_features = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    for df in [x_train, x_test]:
        df.loc[:, categorical_features] = df.loc[:, categorical_features].replace('?', np.nan)
    
    # 构建预处理器，并对数据进行转换
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(x_train)
    X_test_processed = preprocessor.transform(x_test)
    
    # 训练逻辑回归模型
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_processed, y_train.values.ravel())
    
    # 模型预测并评估结果
    y_pred = model.predict(X_test_processed)
    report = classification_report(y_test, y_pred)
    logging.info("分类报告:\n%s", report)
    print(report)

if __name__ == '__main__':
    main()