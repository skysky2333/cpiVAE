# run tuning
python scripts/tune.py --config configs/default.yaml --platform_a data/olink_overlap_train.csv --platform_b data/somascan_overlap_train.csv

# VAE
python scripts/train.py --config configs/default.yaml --platform_a data/olink_overlap_train.csv --platform_b data/somascan_overlap_train.csv  --output_dir outputs_vae

python scripts/impute.py --input_data data/olink_overlap_test.csv --source_platform a --target_platform b --experiment_dir outputs_vae/joint_vae_experiment/version_20250807-225313 --output data/somascan_overlap_test_imputed_vae.csv --output_latent data/olink_overlap_test_latent_vae.csv --output_importance data/somascan_overlap_test_importance_vae.csv --importance_method deeplift
python scripts/impute.py --input_data data/somascan_overlap_test.csv --source_platform b --target_platform a --experiment_dir outputs_vae/joint_vae_experiment/version_20250807-225313 --output data/olink_overlap_test_imputed_vae.csv --output_latent data/somascan_overlap_test_latent_vae.csv --output_importance data/olink_overlap_test_importance_vae.csv --importance_method deeplift

python scripts/confidence.py --input_data data/olink_overlap_test.csv --source_platform a --target_platform b --experiment_dir outputs_vae/joint_vae_experiment/version_20250807-225313 --output data/somascan_overlap_test_confidence_vae.csv --n_runs 100  --method delta
python scripts/confidence.py --input_data data/somascan_overlap_test.csv --source_platform b --target_platform a --experiment_dir outputs_vae/joint_vae_experiment/version_20250807-225313 --output data/olink_overlap_test_confidence_vae.csv --n_runs 100  --method delta
# KNN
python scripts/run_knn_comparison.py --platform_a data/olink_overlap_train.csv --platform_b data/somascan_overlap_train.csv --output_dir outputs_knn --kernel gaussian --platform_impute data/olink_overlap_test.csv --impute_target b
python scripts/run_knn_comparison.py --platform_a data/olink_overlap_train.csv --platform_b data/somascan_overlap_train.csv --output_dir outputs_knn --kernel gaussian --platform_impute data/somascan_overlap_test.csv --impute_target a

cp outputs_knn/olink_overlap_test_cross_imputed_b.csv data/somascan_overlap_test_imputed_knn.csv
cp outputs_knn/somascan_overlap_test_cross_imputed_a.csv data/olink_overlap_test_imputed_knn.csv

# WNN
python scripts/wnn_baseline.py --platform_a data/olink_overlap_train.csv --platform_b data/somascan_overlap_train.csv --output_dir outputs_wnn --platform_impute data/somascan_overlap_test.csv --impute_target a --grid_search
python scripts/wnn_baseline.py --platform_a data/olink_overlap_train.csv --platform_b data/somascan_overlap_train.csv --output_dir outputs_wnn --platform_impute data/olink_overlap_test.csv --impute_target b --grid_search

cp outputs_wnn/olink_overlap_test_cross_imputed_b.csv data/somascan_overlap_test_imputed_wnn.csv
cp outputs_wnn/somascan_overlap_test_cross_imputed_a.csv data/olink_overlap_test_imputed_wnn.csv

# run comparison vae vs rest
python scripts/compare_result.py \
    --truth_a data/olink_overlap_test.csv \
    --truth_b data/somascan_overlap_test.csv \
    --imp_a_m1 data/olink_overlap_test_imputed_vae.csv \
    --imp_a_m2 data/olink_overlap_test_imputed_wnn.csv \
    --imp_b_m1 data/somascan_overlap_test_imputed_vae.csv \
    --imp_b_m2 data/somascan_overlap_test_imputed_wnn.csv \
    --imp_a_m3 data/olink_overlap_test_imputed_knn.csv \
    --imp_a_m4 data/olink_overlap_test_imputed_vae_shuffled.csv \
    --imp_b_m3 data/somascan_overlap_test_imputed_knn.csv \
    --imp_b_m4 data/somascan_overlap_test_imputed_vae_shuffled.csv \
    --method1_name "cpiVAE" \
    --method2_name "WNN" \
    --method3_name "KNN" \
    --method4_name "Permuted" \
    --platform_a_name "Olink" \
    --platform_b_name "Somascan" \
    --transpose \
    --output_dir outputs_comparisons_vae_vs_rest


# latent space analysis
python scripts/latent_space_analysis.py \
    --latent_a data/olink_overlap_test_latent_vae.csv \
    --latent_b data/somascan_overlap_test_latent_vae.csv \
    --truth_a data/olink_overlap_test.csv \
    --truth_b data/somascan_overlap_test.csv \
    --platform_a_name "S->O" \
    --platform_b_name "O->S" \
    --output_dir output_comparisions_latent

# feature importance analysis
python scripts/feature_importance_analysis.py \
    --importance_a_to_b data/somascan_overlap_test_importance_vae.csv \
    --importance_b_to_a data/olink_overlap_test_importance_vae.csv \
    --platform_a_name "Olink" \
    --platform_b_name "Somascan" \
    --output_dir output_comparisions_importance \
    --truth_a data/olink_overlap_test.csv \
    --truth_b data/somascan_overlap_test.csv \
    --imp_a_m1 data/olink_overlap_test_imputed_vae.csv \
    --imp_a_m2 data/olink_overlap_test_imputed_wnn.csv \
    --imp_b_m1 data/somascan_overlap_test_imputed_vae.csv \
    --imp_b_m2 data/somascan_overlap_test_imputed_wnn.csv \
    --network_type directed \
    --network_layout spring \
    --threshold_method absolute_importance \
    --ppi_reference data/9606.protein.links.v12.0_converted.tsv \
    --threshold_params 0.005477

python scripts/feature_importance_analysis_correlation.py \
    --truth_a data/olink_overlap_test.csv \
    --platform_a_name "Olink" \
    --platform_b_name "SomaScan" \
    --ppi_reference data/9606.protein.links.v12.0_converted.tsv \
    --output_dir output_comparisions_network_correlation

# TARGET DENSITY RECOMMENDATION (density 0.0366):
#     Best threshold: --threshold_params 0.005477
#     Achieved density: 0.0351 (diff: 0.0015)
#     Network size: 79,301 edges, 2,125 nodes
    

# confidence analysis
python scripts/confidence_analysis.py \
    --truth_a data/olink_overlap_test.csv \
    --truth_b data/somascan_overlap_test.csv \
    --imputed_a data/olink_overlap_test_imputed_vae.csv \
    --imputed_b data/somascan_overlap_test_imputed_vae.csv \
    --confidence_a data/olink_overlap_test_confidence_vae.csv \
    --confidence_b data/somascan_overlap_test_confidence_vae.csv \
    --platform_a_name "Olink" \
    --platform_b_name "Somascan" \
    --method_name "cpiVAE" \
    --output_dir output_confidence \
    --correlation pearson


# run QC VAE
python scripts/qc.py --data_file data/olink_overlap.csv
python scripts/qc.py --data_file data/somascan_overlap.csv

python scripts/qc_feature.py --data_file data/olink_overlap.csv
python scripts/qc_feature.py --data_file data/somascan_overlap.csv







# train with NC paper data and test with ARIC
# VAE
python scripts/train.py --config configs/default.yaml --platform_a data/NC_ARIC/olink_overlap.csv --platform_b data/NC_ARIC/somascan_overlap.csv  --output_dir outputs_vae_NC_ARIC
python scripts/impute.py --input_data data/NC_ARIC/soma_visit_5_log2_SMP_genename.txt --source_platform b --target_platform a --experiment_dir outputs_vae_NC_ARIC/joint_vae_experiment/version_20250715-124150 --output data/NC_ARIC/olink_visit5_imputed_vae.csv --output_latent data/NC_ARIC/olink_visit5_latent_vae.csv --output_importance data/NC_ARIC/olink_visit5_importance_vae.csv --importance_method deeplift
python scripts/latent_space_analysis_oneplatform.py --latent data/NC_ARIC/olink_visit5_latent_vae.csv --truth data/NC_ARIC/olink_visit5_96.csv --platform_name "Olink" --output_dir output_comparisions_latent_NC_ARIC

# KNN
python scripts/run_knn_comparison.py --platform_a data/NC_ARIC/olink_overlap.csv --platform_b data/NC_ARIC/somascan_overlap.csv --output_dir outputs_knn_NC_ARIC --kernel gaussian --platform_impute data/NC_ARIC/soma_visit_5_log2_SMP_genename.txt --impute_target a
cp outputs_knn_NC_ARIC/soma_visit_5_log2_SMP_genename_cross_imputed_a.txt data/NC_ARIC/olink_visit5_imputed_knn.csv

# WNN
python scripts/wnn_baseline.py --platform_a data/NC_ARIC/olink_overlap.csv --platform_b data/NC_ARIC/somascan_overlap.csv --output_dir outputs_wnn_NC_ARIC --platform_impute data/NC_ARIC/soma_visit_5_log2_SMP_genename.txt --impute_target a
cp outputs_wnn_NC_ARIC/soma_visit_5_log2_SMP_genename_cross_imputed_a.txt data/NC_ARIC/olink_visit5_imputed_wnn.csv

# phenotype discovery
python scripts/pheno_discovery.py --truth_a data/NC_ARIC/olink_visit5_96_withGroups.csv --transpose --phenotype_file data/ARIC/phenotypes/visit5/derive52.csv --output_dir outputs_pheno_discovery_NC_ARIC --gender_col GENDER --age_col V5AGE52

# run comparison vae vs rest
python scripts/compare_result_oneplatform.py \
    --truth_a data/NC_ARIC/olink_visit5_96_withGroups.csv \
    --imp_a_m1 data/NC_ARIC/olink_visit5_imputed_vae.csv \
    --imp_a_m2 data/NC_ARIC/olink_visit5_imputed_wnn.csv \
    --imp_a_m3 data/NC_ARIC/olink_visit5_imputed_knn.csv \
    --imp_a_m4 data/NC_ARIC/olink_visit5_imputed_vae_shuffled.csv \
    --method1_name "cpiVAE" \
    --method2_name "WNN" \
    --method3_name "KNN" \
    --method4_name "Permuted" \
    --platform_a_name "Olink" \
    --transpose \
    --output_dir outputs_comparisons_vae_vs_rest_NC_ARIC \
    --phenotype_file data/ARIC/phenotypes/visit5/derive52.csv \
    --binary_pheno LOOPDIUMDCODE51 CHOLMDCODE52 DIABTS55 \
    --continuous_pheno HDLSIU51 BMI51 LDL51 \
    --gender_col GENDER \
    --age_col V5AGE52


# cis pQTL analysis
python scripts/cis_pQTL_analysis.py \
    --truth data/NC_ARIC/pQTL/cis_results/aric_visit5_cis.olink_visit5_96_idMatched_removeEmpty.all_hits_extracted.sig.tsv \
    --method1 data/NC_ARIC/pQTL/cis_results/aric_visit5_cis.olink_visit5_imputed_vae_idMatched.all_hits_extracted.sig.tsv \
    --method2 data/NC_ARIC/pQTL/cis_results/aric_visit5_cis.olink_visit5_imputed_wnn_idMatched.all_hits_extracted.sig.tsv \
    --method3 data/NC_ARIC/pQTL/cis_results/aric_visit5_cis.olink_visit5_imputed_knn_idMatched.all_hits_extracted.sig.tsv \
    --method4 data/NC_ARIC/pQTL/cis_results/aric_visit5_cis.olink_visit5_imputed_vae_shuffled_idMatched.all_hits_extracted.sig.tsv \
    --method1_name "cpiVAE" \
    --method2_name "WNN" \
    --method3_name "KNN" \
    --method4_name "Permuted" \
    --significance_threshold 1e-06