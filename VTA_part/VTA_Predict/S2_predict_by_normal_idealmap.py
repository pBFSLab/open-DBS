from utils.utils import *
from statsmodels.formula.api import mixedlm
from S1_Func_idealmap_and_similarity_norm import *

if __name__ == '__main__':
    set_environ()
    work_dir = '/path/to/your/workspace/DBS_Open_Project/VTA_FC_predict/predict_func_norm_N14'
    os.makedirs(work_dir, exist_ok=True)

    """
	1. Prepare all subject df information
	"""
    df = pd.read_csv('/path/to/your/workspace/clinical_info/DBS_UPDRSIII_Pre_14.csv')
    print(df.head(), df.columns.tolist())

    formula = 'change_rate_1m ~ patient_similarity + age + C(Sex_bin)'

    # Perform leave-one-subject-out cross-validation
    n_patients = df['Subject'].nunique()
    predictions = np.zeros(n_patients)
    gts = np.zeros(n_patients)
    subs = []
    n_subjects = df['Subject'].unique()
    start_index = 0  # Start index
    for sub in tqdm.tqdm(n_subjects):
        # 1. Leave out the i-th patient
        train_data = df[df['Subject'] != sub]
        test_data = df[df['Subject'] == sub]
        gts[start_index:start_index + len(test_data)] = test_data['change_rate_1m'].values[0]
        subs.append(test_data['Subject'].values[0])

        sub_name = test_data['Subject'].values[0]
        # 2. Weight map
        weight_map(sub_name, work_dir)
        # 3. R map
        R_map(sub_name, work_dir)
        # 4. Ideal map
        ideal_map(work_dir, sub)
        # 5. Calculate similarity
        temp_df = similarity_info(df, work_dir)
        train_data = temp_df[temp_df['Subject'] != sub]
        test_data = temp_df[temp_df['Subject'] == sub]
        # Redefine the model
        model = mixedlm.from_formula(formula, data=train_data, groups=train_data['Subject'])
        result = model.fit()
        # Predict the left-out patient
        predictions[start_index:start_index + len(test_data)] = result.predict(test_data)

        # Update start index
        start_index += len(test_data)

    # predictions now contains the predicted UPDRS-III change rate for each patient
    print('gt:', gts)
    print('preds:', predictions)

    r, p = calculate_corr_pearson(gts, predictions)
    print('Pearson correlation:', r, p)
    r, p = calculate_corr_spearman(gts, predictions)
    print('Spearman correlation:', r, p)

    plot_info = pd.DataFrame()
    plot_info['gt'] = gts
    plot_info['preds'] = predictions
    plot_info['sub'] = subs
    plot_info.to_csv(f"{work_dir}/norm_idealmap_predict_N14.csv", index=False)
