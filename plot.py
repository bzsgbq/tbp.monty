import os
from tbp.monty.frameworks.utils.logging_utils import load_stats, print_overall_stats

pretrain_path = os.path.expanduser("~/tbp/results/monty/pretrained_models/")
pretrained_dict = pretrain_path + "pretrained_ycb_v10/surf_agent_1lm_10similarobj/pretrained/"

log_path = os.path.expanduser("~/tbp/results/monty/projects/evidence_eval_runs/logs/")
exp_name = "randrot_noise_10simobj_surf_agent"
exp_path = log_path + exp_name

train_stats, eval_stats, detailed_stats, lm_models = load_stats(exp_path,
                                                                load_train=False, # doesn't load train csv
                                                                load_eval=True, # loads eval_stats.csv
                                                                load_detailed=True, # doesn't load .json
                                                                load_models=True, # loads .pt models
                                                                pretrained_dict=pretrained_dict,
                                                               )

# print_overall_stats(eval_stats)

from tbp.monty.frameworks.utils.plot_utils import (show_initial_hypotheses,
                                                         plot_evidence_at_step)

episode = 0
lm = 'LM_0'
objects = ['knife', 'fork', 'spoon'] # Up to 4 objects to visualize evidence for
current_evidence_update_threshold = -1
save_fig = True
save_path = exp_path + '/stepwise_examples/'

# [optional] Show initial hypotheses for each point on the object
show_initial_hypotheses(detailed_stats, episode, 'knife', rotation=[120,-90], axis=2,
                        save_fig=save_fig, save_path=save_path)
# Plot the evidence for each hypothesis on each of the objects & show the observations used for updating
for step in range(eval_stats['monty_matching_steps'][episode]):
    plot_evidence_at_step(detailed_stats,
                          lm_models,
                              episode,
                              step,
                              objects,
                              is_surface_sensor=True, # set this to False if not using the surface agent
                              save_fig=save_fig,
                              save_path=save_path)