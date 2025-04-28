import logging
from typing import Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置参数
CONFIG = {
    'data_path': '../../data/California/housing.csv',
    'random_state': 42,
    'test_size': 0.2,
    'numeric_features': ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                        'total_bedrooms', 'population', 'households', 'median_income'],
    'categorical_features': ['ocean_proximity'],
    'target_column': 'median_house_value'
}

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """加载并预处理数据集

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 训练集和测试集的特征和标签
    """
    try:
        data_path = Path(CONFIG['data_path'])
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        data = pd.read_csv(data_path)
        X = data.drop(CONFIG['target_column'], axis=1)
        y = data[CONFIG['target_column']]
        X.replace('', np.nan, inplace=True)
        
        return train_test_split(
            X, y, 
            test_size=CONFIG['test_size'], 
            random_state=CONFIG['random_state']
        )
    except Exception as e:
        logger.error(f"加载数据时发生错误: {str(e)}")
        raise

def build_model() -> Pipeline:
    """构建机器学习pipeline

    Returns:
        Pipeline: 预处理和模型的pipeline
    """
    try:
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('power_transformer', PowerTransformer(method='yeo-johnson')),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer([
            ('num', num_transformer, CONFIG['numeric_features']),
            ('cat', cat_transformer, CONFIG['categorical_features'])
        ])

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', SVR(
            kernel='rbf',
            C=1.0,
            epsilon=0.1
            ))
        ])
        return model
    except Exception as e:
        logger.error(f"构建模型时发生错误: {str(e)}")
        raise

def create_search_cv(model: Pipeline) -> RandomizedSearchCV:
    """创建随机搜索交叉验证对象

    Args:
        model (Pipeline): 待优化的模型pipeline

    Returns:
        RandomizedSearchCV: 随机搜索对象
    """
    param_distributions = {
        'regressor__C': loguniform(0.1, 100),
        'regressor__epsilon': loguniform(0.01, 1),
        'regressor__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'regressor__degree': randint(2, 5),  # Only relevant for 'poly' kernel
        'regressor__gamma': ['scale', 'auto'] + list(loguniform(0.001, 1).rvs(10))  # For 'rbf', 'poly', 'sigmoid'
    }
    
    return RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring='r2',
        random_state=CONFIG['random_state'],
        n_jobs=-1
    )

def main():
    """主函数：执行模型训练和评估流程"""
    try:
        logger.info("开始加载数据...")
        X_train, X_test, y_train, y_test = load_data()
        
        logger.info("构建模型...")
        model = build_model()
        
        logger.info("开始随机搜索超参数...")
        search_cv = create_search_cv(model)
        search_cv.fit(X_train, y_train)
        
        final_score = search_cv.score(X_test, y_test)
        logger.info(f"最终测试集R2得分: {final_score:.3f}")
        
        logger.info("最佳参数:")
        for param, value in search_cv.best_params_.items():
            logger.info(f"{param}: {value}")
            
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()

#最终测试集R2得分: 0.544