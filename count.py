import pandas as pd

# 加载数据
df = pd.read_csv('master.csv')

# 定义一个函数来计算每个fold中train和test的morphology计数
def count_morphology_per_fold(df, fold_col):
    # 筛选出该fold列为'train'或'test'的行
    train_df = df[df[fold_col] == 'train']
    test_df = df[df[fold_col] == 'test']
    
    # 计算train中的morphology计数
    train_counts = train_df['morphology'].value_counts().reset_index()
    train_counts.columns = ['Morphology', 'Count']
    train_counts['Fold'] = fold_col
    train_counts['Type'] = 'Train'
    
    # 计算test中的morphology计数
    test_counts = test_df['morphology'].value_counts().reset_index()
    test_counts.columns = ['Morphology', 'Count']
    test_counts['Fold'] = fold_col
    test_counts['Type'] = 'Test'
    
    # 合并train和test的计数结果
    return pd.concat([train_counts, test_counts])

# 逐个处理每个fold
all_folds_counts = []
for fold in ['set0', 'set1', 'set2', 'set3', 'set4']:
    fold_counts = count_morphology_per_fold(df, fold)
    all_folds_counts.append(fold_counts)

# 合并所有fold的计数结果
final_counts = pd.concat(all_folds_counts)

# 显示结果
print(final_counts)
