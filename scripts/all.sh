# main settings
python main.py --config_path settings/main/v1_PPO-h200-c02-n64-b32-R15-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/main/v2_PPO-h400-c02-n64-b32-R15-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/main/v4_PPO-h1600-c02-n64-b32-R15-atten1FT64T-k1-rA.yaml


# attention
python main.py --config_path settings/attention/v1_PPO-h200-c02-n64-b32-R15-atten1FT64F-k1-rA.yaml
python main.py --config_path settings/attention/v1_PPO-h200-c02-n64-b32-R15-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/attention/v1_PPO-h200-c02-n64-b32-R15-atten1TT64F-k1-rA.yaml
python main.py --config_path settings/attention/v1_PPO-h200-c02-n64-b32-R15-atten1TT64T-k1-rA.yaml
python main.py --config_path settings/attention/v1_PPO-h200-c02-n64-b32-R15-atten1FF256F-k1-rA.yaml
python main.py --config_path settings/attention/v1_PPO-h200-c02-n64-b32-R15-atten1FF256T-k1-rA.yaml
python main.py --config_path settings/attention/v1_PPO-h200-c02-n64-b32-R15-atten1TF256F-k1-rA.yaml
python main.py --config_path settings/attention/v1_PPO-h200-c02-n64-b32-R15-atten1TF256T-k1-rA.yaml

python main.py --config_path settings/attention/v2_PPO-h400-c02-n64-b32-R15-atten1FT64F-k1-rA.yaml
python main.py --config_path settings/attention/v2_PPO-h400-c02-n64-b32-R15-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/attention/v2_PPO-h400-c02-n64-b32-R15-atten1TT64F-k1-rA.yaml
python main.py --config_path settings/attention/v2_PPO-h400-c02-n64-b32-R15-atten1TT64T-k1-rA.yaml
python main.py --config_path settings/attention/v2_PPO-h400-c02-n64-b32-R15-atten1FF256F-k1-rA.yaml
python main.py --config_path settings/attention/v2_PPO-h400-c02-n64-b32-R15-atten1FF256T-k1-rA.yaml
python main.py --config_path settings/attention/v2_PPO-h400-c02-n64-b32-R15-atten1TF256F-k1-rA.yaml
python main.py --config_path settings/attention/v2_PPO-h400-c02-n64-b32-R15-atten1TF256T-k1-rA.yaml

python main.py --config_path settings/attention/v3_PPO-h1600-c02-n64-b32-R15-atten1FT64F-k1-rA.yaml
python main.py --config_path settings/attention/v3_PPO-h1600-c02-n64-b32-R15-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/attention/v3_PPO-h1600-c02-n64-b32-R15-atten1TT64F-k1-rA.yaml
python main.py --config_path settings/attention/v3_PPO-h1600-c02-n64-b32-R15-atten1TT64T-k1-rA.yaml
python main.py --config_path settings/attention/v3_PPO-h1600-c02-n64-b32-R15-atten1FF256F-k1-rA.yaml
python main.py --config_path settings/attention/v3_PPO-h1600-c02-n64-b32-R15-atten1FF256T-k1-rA.yaml
python main.py --config_path settings/attention/v3_PPO-h1600-c02-n64-b32-R15-atten1TF256F-k1-rA.yaml
python main.py --config_path settings/attention/v3_PPO-h1600-c02-n64-b32-R15-atten1TF256T-k1-rA.yaml

python main.py --config_path settings/attention/v4_PPO-h1600-c02-n64-b32-R15-atten1FT64F-k1-rA.yaml
python main.py --config_path settings/attention/v4_PPO-h1600-c02-n64-b32-R15-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/attention/v4_PPO-h1600-c02-n64-b32-R15-atten1TT64F-k1-rA.yaml
python main.py --config_path settings/attention/v4_PPO-h1600-c02-n64-b32-R15-atten1TT64T-k1-rA.yaml
python main.py --config_path settings/attention/v4_PPO-h1600-c02-n64-b32-R15-atten1FF256F-k1-rA.yaml
python main.py --config_path settings/attention/v4_PPO-h1600-c02-n64-b32-R15-atten1FF256T-k1-rA.yaml
python main.py --config_path settings/attention/v4_PPO-h1600-c02-n64-b32-R15-atten1TF256F-k1-rA.yaml
python main.py --config_path settings/attention/v4_PPO-h1600-c02-n64-b32-R15-atten1TF256T-k1-rA.yaml


# ablation study / use original network of reference paper
python main.py --config_path settings/ablation/orig_net/v1_PPO-h200-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/ablation/orig_net/v2_PPO-h400-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/ablation/orig_net/v3_PPO-h1600-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/ablation/orig_net/v4_PPO-h1600-c02-n64-b32-R15-k1-rA.yaml


# ablation study / use ACKTR to update policy (orig_net)
python main.py --config_path settings/ablation/ACKTR/v1_ACKTR-h200-n64-R15-rA.yaml
python main.py --config_path settings/ablation/ACKTR/v2_ACKTR-h400-n64-R15-rA.yaml
python main.py --config_path settings/ablation/ACKTR/v3_ACKTR-h1600-n64-R15-rA.yaml
python main.py --config_path settings/ablation/ACKTR/v4_ACKTR-h1600-n64-R15-rA.yaml


# ablation study / compare diff reward type
python main.py --config_path settings/ablation/diff_reward_type/v1_PPO-h200-c02-n64-b32-R15-atten1FT64T-k1-rC.yaml
python main.py --config_path settings/ablation/diff_reward_type/v2_PPO-h400-c02-n64-b32-R15-atten1FT64T-k1-rC.yaml
python main.py --config_path settings/ablation/diff_reward_type/v3_PPO-h1600-c02-n64-b32-R15-atten1FT64T-k1-rC.yaml
python main.py --config_path settings/ablation/diff_reward_type/v4_PPO-h1600-c02-n64-b32-R15-atten1FT64T-k1-rC.yaml


# ablation study / use prediction mask to train
python main.py --config_path settings/ablation/pred_mask/v1_PPO-h200-c02-n64-b32-R15-atten1FT64T-k1-rA-P.yaml
python main.py --config_path settings/ablation/pred_mask/v2_PPO-h400-c02-n64-b32-R15-atten1FT64T-k1-rA-P.yaml
python main.py --config_path settings/ablation/pred_mask/v3_PPO-h1600-c02-n64-b32-R15-atten1FT64T-k1-rA-P.yaml
python main.py --config_path settings/ablation/pred_mask/v4_PPO-h1600-c02-n64-b32-R15-atten1FT64T-k1-rA-P.yaml


# sensitive study / replace mask diff coef fo v1
python main.py --config_path settings/sensitive/mask_replace/v1_PPO-h200-c02-n64-b32-R0-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v1_PPO-h200-c02-n64-b32-R7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v1_PPO-h200-c02-n64-b32-R30-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v1_PPO-h200-c02-n64-b32-R50-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v1_PPO-h200-c02-n64-b32-R100-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v1_PPO-h200-c02-n64-b32-R500-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v1_PPO-h200-c02-n64-b32-Re3-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v1_PPO-h200-c02-n64-b32-Re4-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v1_PPO-h200-c02-n64-b32-Re5-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v1_PPO-h200-c02-n64-b32-Re6-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v1_PPO-h200-c02-n64-b32-Re7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v1_PPO-h200-c02-n64-b32-Re8-atten1FT64T-k1-rA.yaml


# sensitive study / minus mask diff coef fo v1
python main.py --config_path settings/sensitive/mask_minus/v1_PPO-h200-c02-n64-b32-M0-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v1_PPO-h200-c02-n64-b32-M7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v1_PPO-h200-c02-n64-b32-M15-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v1_PPO-h200-c02-n64-b32-M30-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v1_PPO-h200-c02-n64-b32-M50-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v1_PPO-h200-c02-n64-b32-M100-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v1_PPO-h200-c02-n64-b32-M500-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v1_PPO-h200-c02-n64-b32-Me3-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v1_PPO-h200-c02-n64-b32-Me4-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v1_PPO-h200-c02-n64-b32-Me5-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v1_PPO-h200-c02-n64-b32-Me6-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v1_PPO-h200-c02-n64-b32-Me7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v1_PPO-h200-c02-n64-b32-Me8-atten1FT64T-k1-rA.yaml


# sensitive study / replace mask diff coef fo v2
python main.py --config_path settings/sensitive/mask_replace/v2_PPO-h400-c02-n64-b32-R0-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v2_PPO-h400-c02-n64-b32-R7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v2_PPO-h400-c02-n64-b32-R30-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v2_PPO-h400-c02-n64-b32-R50-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v2_PPO-h400-c02-n64-b32-R100-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v2_PPO-h400-c02-n64-b32-R500-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v2_PPO-h400-c02-n64-b32-Re3-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v2_PPO-h400-c02-n64-b32-Re4-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v2_PPO-h400-c02-n64-b32-Re5-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v2_PPO-h400-c02-n64-b32-Re6-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v2_PPO-h400-c02-n64-b32-Re7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v2_PPO-h400-c02-n64-b32-Re8-atten1FT64T-k1-rA.yaml


# sensitive study / minus mask diff coef fo v2
python main.py --config_path settings/sensitive/mask_minus/v2_PPO-h400-c02-n64-b32-M0-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v2_PPO-h400-c02-n64-b32-M7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v2_PPO-h400-c02-n64-b32-M15-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v2_PPO-h400-c02-n64-b32-M30-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v2_PPO-h400-c02-n64-b32-M50-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v2_PPO-h400-c02-n64-b32-M100-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v2_PPO-h400-c02-n64-b32-M500-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v2_PPO-h400-c02-n64-b32-Me3-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v2_PPO-h400-c02-n64-b32-Me4-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v2_PPO-h400-c02-n64-b32-Me5-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v2_PPO-h400-c02-n64-b32-Me6-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v2_PPO-h400-c02-n64-b32-Me7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v2_PPO-h400-c02-n64-b32-Me8-atten1FT64T-k1-rA.yaml


# sensitive study / replace mask diff coef fo v3
python main.py --config_path settings/sensitive/mask_replace/v3_PPO-h1600-c02-n64-b32-R0-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v3_PPO-h1600-c02-n64-b32-R7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v3_PPO-h1600-c02-n64-b32-R30-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v3_PPO-h1600-c02-n64-b32-R50-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v3_PPO-h1600-c02-n64-b32-R100-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v3_PPO-h1600-c02-n64-b32-R500-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v3_PPO-h1600-c02-n64-b32-Re3-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v3_PPO-h1600-c02-n64-b32-Re4-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v3_PPO-h1600-c02-n64-b32-Re5-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v3_PPO-h1600-c02-n64-b32-Re6-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v3_PPO-h1600-c02-n64-b32-Re7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v3_PPO-h1600-c02-n64-b32-Re8-atten1FT64T-k1-rA.yaml


# sensitive study / minus mask diff coef fo v3
python main.py --config_path settings/sensitive/mask_minus/v3_PPO-h1600-c02-n64-b32-M0-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v3_PPO-h1600-c02-n64-b32-M7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v3_PPO-h1600-c02-n64-b32-M15-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v3_PPO-h1600-c02-n64-b32-M30-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v3_PPO-h1600-c02-n64-b32-M50-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v3_PPO-h1600-c02-n64-b32-M100-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v3_PPO-h1600-c02-n64-b32-M500-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v3_PPO-h1600-c02-n64-b32-Me3-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v3_PPO-h1600-c02-n64-b32-Me4-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v3_PPO-h1600-c02-n64-b32-Me5-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v3_PPO-h1600-c02-n64-b32-Me6-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v3_PPO-h1600-c02-n64-b32-Me7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v3_PPO-h1600-c02-n64-b32-Me8-atten1FT64T-k1-rA.yaml


# sensitive study / replace mask diff coef fo v4
python main.py --config_path settings/sensitive/mask_replace/v4_PPO-h1600-c02-n64-b32-R0-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v4_PPO-h1600-c02-n64-b32-R7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v4_PPO-h1600-c02-n64-b32-R30-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v4_PPO-h1600-c02-n64-b32-R50-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v4_PPO-h1600-c02-n64-b32-R100-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v4_PPO-h1600-c02-n64-b32-R500-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v4_PPO-h1600-c02-n64-b32-Re3-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v4_PPO-h1600-c02-n64-b32-Re4-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v4_PPO-h1600-c02-n64-b32-Re5-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v4_PPO-h1600-c02-n64-b32-Re6-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v4_PPO-h1600-c02-n64-b32-Re7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_replace/v4_PPO-h1600-c02-n64-b32-Re8-atten1FT64T-k1-rA.yaml


# sensitive study / minus mask diff coef fo v4
python main.py --config_path settings/sensitive/mask_minus/v4_PPO-h1600-c02-n64-b32-M0-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v4_PPO-h1600-c02-n64-b32-M7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v4_PPO-h1600-c02-n64-b32-M15-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v4_PPO-h1600-c02-n64-b32-M30-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v4_PPO-h1600-c02-n64-b32-M50-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v4_PPO-h1600-c02-n64-b32-M100-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v4_PPO-h1600-c02-n64-b32-M500-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v4_PPO-h1600-c02-n64-b32-Me3-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v4_PPO-h1600-c02-n64-b32-Me4-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v4_PPO-h1600-c02-n64-b32-Me5-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v4_PPO-h1600-c02-n64-b32-Me6-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v4_PPO-h1600-c02-n64-b32-Me7-atten1FT64T-k1-rA.yaml
python main.py --config_path settings/sensitive/mask_minus/v4_PPO-h1600-c02-n64-b32-Me8-atten1FT64T-k1-rA.yaml


# sensitive study / diff PPO's hyperparameters for v1
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h100-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h100-c02-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h100-c02-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h100-c02-n128-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h100-c04-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h100-c04-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h100-c04-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h100-c04-n128-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h200-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h200-c02-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h200-c02-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h200-c02-n128-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h200-c04-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h200-c04-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h200-c04-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v1/v1_PPO-h200-c04-n128-b64-R15-k5-rA.yaml


# sensitive study / diff PPO's hyperparameters for v2
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h400-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h400-c02-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h400-c02-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h400-c02-n128-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h400-c04-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h400-c04-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h400-c04-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h400-c04-n128-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h800-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h800-c02-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h800-c02-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h800-c02-n128-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h800-c04-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h800-c04-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h800-c04-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v2/v2_PPO-h800-c04-n128-b64-R15-k5-rA.yaml


# sensitive study / diff PPO's hyperparameters for v3
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h1600-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h1600-c02-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h1600-c02-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h1600-c02-n128-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h1600-c04-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h1600-c04-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h1600-c04-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h1600-c04-n128-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h3200-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h3200-c02-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h3200-c02-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h3200-c02-n128-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h3200-c04-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h3200-c04-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h3200-c04-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v3/v3_PPO-h3200-c04-n128-b64-R15-k5-rA.yaml


# sensitive study / diff PPO's hyperparameters for v4
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h1600-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h1600-c02-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h1600-c02-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h1600-c02-n128-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h1600-c04-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h1600-c04-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h1600-c04-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h1600-c04-n128-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h3200-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h3200-c02-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h3200-c02-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h3200-c02-n128-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h3200-c04-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h3200-c04-n64-b64-R15-k5-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h3200-c04-n128-b32-R15-k1-rA.yaml
python main.py --config_path settings/sensitive/hyper_v4/v4_PPO-h3200-c04-n128-b64-R15-k5-rA.yaml
